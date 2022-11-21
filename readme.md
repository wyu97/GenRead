## Code for GenRead: Genrate rather than Retrieve!

## Introduction

-- This is the official implementation of our *pre-print* paper "**Generate rather than Retrieve: Large Language Models are Strong Context Generators**" [\[arXiv\]](https://arxiv.org/abs/2209.10063).

**We will add the code (Generate-then-Read) and the trained checkpoint (FiD) used to reproduce our experiments these days. If you want to get timely updates, please click "Watch".**

## Experimental Setup


-- Create an environment and install openai package via `pip install openai`.

-- Add your OpenAI API key at line 12 `openai.api_key` in `inference.py`

## Download the Dataset

From Google drive [\[link\]](https://drive.google.com/drive/folders/1lFFTklW_0HuR53hLpFdLClgfSAhXn_2f?usp=sharing)

## Download our Outputs

-- If you have limited access to OpenAI API, you could directly use our outputs, without spending money on reproducing our experiments. 

From Google drive [\[step1\]](https://drive.google.com/drive/folders/1u7VUOX2l86g4JkMPxPZ1vhMW8O7mwRZw?usp=sharing) [\[step1\]](https://drive.google.com/drive/folders/1s5chlju2Nzh4IqH1I49m73mwlnVL2318?usp=sharing)


## Run the Code

-- Step1: generate background document 

`python mainfunc.py --dataset {dataset} --task step1 --split test`

-- Step2: infer answers from the document

`python mainfunc.py --dataset {dataset} --task step2 --split test`

## Citation

```
@article{yu2022generate,
  title={Generate rather than retrieve: Large language models are strong context generators},
  author={Yu, Wenhao and Iter, Dan and Wang, Shuohang and Xu, Yichong and Ju, Mingxuan and Sanyal, Soumya and Zhu, Chenguang and Zeng, Michael and Jiang, Meng},
  journal={arXiv preprint arXiv:2209.10063},
  year={2022}
}
```

Please kindly cite our paper if you find this paper and the codes helpful.
