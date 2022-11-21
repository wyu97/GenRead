import argparse
import os
import json

from inference import run_main
from evaluation import (
    eval_recall,
    eval_question_answering,
    eval_fact_checking,
    eval_dialogue_system
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

    return lines


def step1(dataset, datatype, split, max_tokens, engine, prompt, pid, n, temp):

    inputfile = f'indatasets/{dataset}/{dataset}-{split}.jsonl'
    inlines = readfiles(inputfile)

    if (temp is None) or (temp == 0):
        outputfolder = f'backgrounds-greedy-{engine}/{dataset}'
    else: # tempature > 0
        outputfolder = f'backgrounds-sample(n={n},temp={temp})-{engine}/{dataset}'
    os.makedirs(outputfolder, exist_ok=True)
    outputfile = f'{outputfolder}/{dataset}-{split}-p{pid}.jsonl'
    
    run_main(inlines, outputfile, engine, prompt, max_tokens, n, temp)

    if datatype == 'question answering': ## Eval Recall@K score
        recallfile = f'{outputfolder}/{dataset}-recall@k.jsonl'
        with open(recallfile, 'a') as recallout:
            recall, length = eval_recall(outputfile)
            outmetrics = {
                'outputfile': outputfile,
                'prompt': prompt,
                'recall@k': recall,
                'length': length,
            }
            print(f'Recall@k: {recall}; Avg.Length: {length}')
            recallout.write(json.dumps(outmetrics) + '\n')


def step2(dataset, datatype, split, max_tokens, engine, prompt, pid):

    inputfile = f'backgrounds-greedy-{engine}/{dataset}/{dataset}-{split}-p{pid}.jsonl'
    inlines = readfiles(inputfile)

    outputfolder = f'finaloutput-greedy-{engine}/{dataset}'
    os.makedirs(outputfolder, exist_ok=True)
    outputfile = f'{outputfolder}/{dataset}-{split}-p{pid}.jsonl'
    
    run_main(inlines, outputfile, engine, prompt, max_tokens)

    if datatype == 'question answering': ## Eval Exact Match
        evalfile = f'{outputfolder}/{dataset}-metrics.jsonl'
        with open(evalfile, 'a') as evalout:
            emscore, length = eval_question_answering(outputfile)
            outmetrics = {
                'outputfile': outputfile,
                'prompt': prompt,
                'exact match': emscore,
                'length': length,
            }
            print(f'Exact Match: {emscore}; Avg.Length: {length}')
            evalout.write(json.dumps(outmetrics) + '\n')
    
    elif datatype == 'fact checking': ## Eval Accuracy
        evalfile = f'{outputfolder}/{dataset}-metrics.jsonl'
        with open(evalfile, 'a') as evalout:
            accuracy, length = eval_fact_checking(outputfile)
            outmetrics = {
                'outputfile': outputfile,
                'prompt': prompt,
                'accuracy': accuracy,
                'length': length,
            }
            print(f'Accuracy: {accuracy}; Avg.Length: {length}')
            evalout.write(json.dumps(outmetrics) + '\n')

    elif datatype == 'dialogue system': ## Eval F1 and Rouge
        evalfile = f'{outputfolder}/{dataset}-metrics.jsonl'
        with open(evalfile, 'a') as evalout:
            f1score, rougel, length = eval_dialogue_system(outputfile)
            outmetrics = {
                'outputfile': outputfile,
                'prompt': prompt,
                'f1-score': f1score,
                'rouge-l': rougel,
                'length': length,
            }
            print(f'F1-score: {f1score}; Rouge-L: {rougel}; Avg.Length: {length}')
            evalout.write(json.dumps(outmetrics) + '\n')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    # Required parameters
    parser.add_argument("--dataset", default=None, type=str, required=True,
        help="The output directory where the model checkpoints and predictions will be written.",
    )
    parser.add_argument("--task", default=None, type=str, required=True,
        help="The output directory where the model checkpoints and predictions will be written.",
    )
    parser.add_argument("--split", default=None, type=str, required=True,
        help="The output directory where the model checkpoints and predictions will be written.",
    )
    parser.add_argument("--engine", default='text-davinci-002', type=str, required=False,
        help="The output directory where the model checkpoints and predictions will be written.",
    )
    parser.add_argument("--num_sequence", default=1, type=int, required=False,
        help="The output directory where the model checkpoints and predictions will be written.",
    )
    parser.add_argument("--temperature", default=0, type=float, required=False,
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

    if args.task == 'step1':
        max_tokens = 300
    elif args.task == 'step2':
        if datatype == 'dialogue system':
            max_tokens = 50
        else: # QA and Fact ...
            max_tokens = 10

    promptlines = open(f'inprompts/regular_prompts.jsonl', 'r').readlines()
    for line in promptlines:
        line = json.loads(line)

        if line['type'] == datatype and line['task'] == args.task:
            prompt = line['prompt']
            pid = line['pid']

            if args.task == 'step1':
                outputs = step1(args.dataset, datatype, args.split, max_tokens, args.engine, 
                    prompt, pid, args.num_sequence, args.temperature)

            elif args.task == 'step2':
                outputs = step2(args.dataset, datatype, args.split, 
                    max_tokens, args.engine, prompt, pid)

            break ## only use the first prompt 