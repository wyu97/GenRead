import openai
import os
import time
import threading
import json
import _thread
from tqdm import tqdm

from contextlib import contextmanager
from collections import defaultdict

openai.api_key = "your openai api key"

class TimeoutException(Exception):
    def __init__(self, msg=''):
        self.msg = msg


@contextmanager
def time_limit(seconds, msg=''):
    
    timer = threading.Timer(seconds, lambda: _thread.interrupt_main())
    timer.start()
    try:
        yield
    except KeyboardInterrupt:
        raise TimeoutException("Timed out for operation {}".format(msg))
    finally:
        # if the action ends in specified time, timer is canceled
        timer.cancel()


def add_prompt(item, prompt):

    def rmreturn(s):
        s = s.replace('\n\n', ' ')
        s = s.replace('\n', ' ')
        return s.strip()

    query = item['question']
    prompt = prompt.replace('{query}', query)

    if item.get('output'): # background info
        backinfo = rmreturn(item['output'][0])
        prompt = prompt.replace('{background}', backinfo)

    return prompt


def clustering_prompt(items, prompt):

    def rmreturn(s):
        s = s.replace('\n\n', ' ')
        s = s.replace('\n', ' ')
        return s.strip()

    cluster_prompts = []
    for item in items:
        query = item['question']
        backinfo = rmreturn(item['output'][0])
        item_prompt = prompt.replace('{query}', query)
        item_prompt += f' {backinfo}'
        cluster_prompts.append(item_prompt)

    cluster_prompts.append(prompt)
    return ' \n\n\n\n '.join(cluster_prompts)


def run_embeddings(input_text, engine='text-similarity-davinci-001'):
    
    texts = [t.replace('\n', '') for t in input_text]
    outputs = openai.Embedding.create(input=texts, model=engine)['data']
    embeddings = [o['embedding'] for o in outputs]

    return embeddings


def run_inference(inputs_with_prompts, engine, max_tokens, num_sequence=1, temp=0):

    completions = {"choices": []}
    for _ in range(200):
        try:
            with time_limit(20, 'run gpt-3'):
                completions = openai.Completion.create(
                    engine=engine, 
                    max_tokens=max_tokens, 
                    prompt=inputs_with_prompts, 
                    temperature=temp, 
                    n=num_sequence, # num of returned sequence
                    )
                break
        except:
                time.sleep(2)

    outputs = [c["text"] for c in completions["choices"]]
    return outputs


def run_main(inlines, outfile, engine, prompt, max_tokens, n=1, temp=0):

    if os.path.exists(outfile):
        outs = open(outfile, 'a', encoding='utf8')
        num_lines = len(open(outfile, 'r').readlines())
        inlines = inlines[num_lines - 1: ]
    else: # not os.path.exists(outfile)
        outs = open(outfile, 'a', encoding='utf8')
        outs.write(json.dumps({"prompt": prompt}) + '\n')

    pbar = tqdm(total = len(inlines))
    index = 0
    pbar.update(index)
    while index < len(inlines):
        inputs, answers = [], []
        inputs_with_prompts = []
        for _ in range(20):
            if index >= len(inlines): break
            input_with_prompt = add_prompt(inlines[index], prompt)
            inputs.append(inlines[index]['question']) ## a string
            answers.append(inlines[index]['answer']) ## a list of strings
            inputs_with_prompts.append(input_with_prompt)
            index += 1

        samples = defaultdict(list)
        outputs = run_inference(inputs_with_prompts, 
            engine, max_tokens, n, temp)
        for j, output in enumerate(outputs):
            samples[j//n].append(output)

        for i in range(len(inputs_with_prompts)):
            outs.write(json.dumps({
                'question': inputs[i], 
                'answer': answers[i], 
                'output': samples[i]}) 
                +'\n')

        pbar.update(len(inputs_with_prompts))

    pbar.close()
    outs.close()