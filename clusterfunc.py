import argparse
import os
import json
import random
from tqdm import tqdm
import numpy as np
from sklearn.cluster import KMeans 

from collections import defaultdict
from evaluation import has_answer
from inference import (
    run_embeddings, 
    clustering_prompt
)


def readfiles(infile):

    if infile.endswith('json'): 
        lines = json.load(open(infile, 'r', encoding='utf8'))
    elif infile.endswith('jsonl'): 
        lines = open(infile, 'r', encoding='utf8').readlines()
        lines = [json.loads(l) for l in lines]
    else:
        raise NotImplementedError

    if len(lines[0]) == 1 and lines[0].get('prompt'): 
        lines = lines[1:] ## skip prompt line

    return lines[:200]


''' step1: generate embeddings for each question-document pair'''
def step1(infile, outfile, engine='text-similarity-davinci-001'):

    inlines = readfiles(infile)

    kept_idx = []
    for idx, line in enumerate(inlines):

        answers = line['answer']
        passage = line['output'][0]
        
        if has_answer(answers, passage):
            kept_idx.append(idx)

    inlines = [l for i, l in enumerate(inlines) if i in kept_idx]
    print(f'number of lines: {len(inlines)}')

    if os.path.exists(outfile):
        with open(outfile, 'r') as f:
            num_lines = len(f.readlines())
        outfile = open(outfile, 'a', encoding='utf8')
        inlines = inlines[num_lines: ]
    else: # not os.path.exists(outfile)
        outfile = open(outfile, 'a', encoding='utf8')

    ## generate embeddings by batch
    random.shuffle(inlines)
    pbar = tqdm(total = len(inlines))
    index = 0
    pbar.update(index)
    while index < len(inlines):
        inputs, emb_inputs = [], []

        for _ in range(20):
            if index >= len(inlines): break

            line = inlines[index]
            inputs.append(line)
            question = line['question']
            passage = line['output'][0]
            emb_input = ' '.join([question, passage])
            emb_inputs.append(emb_input)
            index += 1

        emebddings = run_embeddings(emb_inputs, engine)
        for line, emb in zip(inputs, emebddings):
            line['embedding'] = emb
            outfile.write(json.dumps(line) + '\n')

        pbar.update(20)

    pbar.close()
    outfile.close()


''' step2: K-means clustering '''
def step2(infile, outfile):

    inlines = [json.loads(l) for l in open(infile, 'r').readlines()]
    matrix = np.vstack([l['embedding'] for l in inlines])
    print(f'embedding matrix: {matrix.shape}')
    n_clusters = 20
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    kmeans.fit(matrix)
    labels = kmeans.labels_

    assert len(inlines) == len(labels)
    for line, label in zip(inlines, labels):
        line['label'] = str(label)
        del line['embedding']

    with open(outfile, 'w') as outfile: 
        for line in inlines:
            outfile.write(json.dumps(line) + '\n')


''' step3: sample in-context demonstrations '''
def step3(infile, outfile, prompt):

    inlines = [json.loads(l) for l in open(infile, 'r').readlines()]

    cluster2examples = defaultdict(list)
    for _, line in enumerate(inlines):
        clusterid = line['label']
        cluster2examples[clusterid].append(line)

    with open(outfile, 'w') as outfile:
        for cid, ls in cluster2examples.items():
            random.shuffle(ls)
            cluster_prompt = clustering_prompt(ls[:5], prompt)
            outfile.write(json.dumps({
                'type': 'question answering',
                'task': 'step1',
                'pid': f'c-{cid}',
                'prompt': cluster_prompt,
            }) + '\n')


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    
    # Required parameters
    parser.add_argument("--dataset", default=None, type=str, required=True,
        help="The output directory where the model checkpoints and predictions will be written.",
    )
    parser.add_argument("--engine", default='text-davinci-002', type=str, required=False,
        help="The output directory where the model checkpoints and predictions will be written.",
    )
    parser.add_argument("--pid", default='1', type=str, required=False,
        help="The output directory where the model checkpoints and predictions will be written.",
    )

    args = parser.parse_args()

    if args.dataset in ['nq', 'webq', 'tqa', 'twiki']:
        datatype = 'question answering'
    elif args.dataset in ['fever', 'fm2']:
        datatype = 'fact checking'
    elif args.dataset in ['wow']: 
        datatype = 'dialogue system'
    else: # other task type?
        raise NotImplementedError

    infolder = f'backgrounds-greedy-{args.engine}/{args.dataset}'
    infile = f'{infolder}/{args.dataset}-train-p{args.pid}.jsonl'
    outfolder = f'embeddings-greedy-{args.engine}/{args.dataset}'
    os.makedirs(outfolder, exist_ok=True)
    embfile = f'{outfolder}/{args.dataset}-train-embeddings.jsonl'
    clsfile = f'{outfolder}/{args.dataset}-train-clusters.jsonl'
    promptfile = f'{outfolder}/{args.dataset}-cluster-prompts.jsonl'

    step1(infile, embfile) # step1: generate embeddings
    step2(embfile, clsfile) # step2: k-means cluster

    promptlines = open(f'inprompts/regular_prompts.jsonl', 'r').readlines()
    for line in promptlines:
        line = json.loads(line)

        if line['type'] == datatype and line['task'] == 'step1':
            prompt = line['prompt']
            step3(clsfile, promptfile, prompt)
            break ## only use the first prompt 