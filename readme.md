## Code for GenRead: Genrate rather than Retrieve!

### Introduction & Setup

-- This is the official implementation of our *pre-print* paper **"Generate rather than Retrieve: Large Language Models are Strong Context Generators"** [\[arXiv\]](https://arxiv.org/abs/2209.10063).

-- Create an environment and install openai package via `pip install openai`.

-- Add your OpenAI API key at `openai.api_key` (line 12) in `inference.py`

### Download the Datasets

-- From their official websites: [\[NQ/TriviaQA/WebQ\]](https://github.com/facebookresearch/DPR) / [\[FM2\]](https://github.com/google-research/fool-me-twice) / [\[FEVER/Wizard\]](https://github.com/facebookresearch/KILT)

-- From Google drive: (we unified the formats of the above datasets) [\[link\]](https://drive.google.com/drive/folders/1lFFTklW_0HuR53hLpFdLClgfSAhXn_2f?usp=sharing)

-- (Please put them into `indataset` folder. Now it contains `webq` and `fm2`.)

### Zero-shot Setting 

**Step1: generate background document.**

```
python mainfunc.py 
  --dataset {dataset} 
  --task step1 
  --split test
```

-- Note: we use the `text-davinci-002` in our experiment; we use greedy search in the zero-shot setting, to ensure the reproducibility of our experiments. 

-- Note: if you have limited access to OpenAI API, you could directly use our outputs, without spending money on reproducing our experiments. [\[zero-shot: step1\]](https://drive.google.com/drive/folders/1u7VUOX2l86g4JkMPxPZ1vhMW8O7mwRZw?usp=sharing)

**Step2: infer answer from document.**

-- Trick: we remove the `\n` in the generated documents. 

-- Note: if you have limited access to OpenAI API, you could directly use our outputs, without spending money on reproducing our experiments. [\[zero-shot: step2\]](https://drive.google.com/drive/folders/1s5chlju2Nzh4IqH1I49m73mwlnVL2318?usp=sharing)


### Supervised Setting 

**Method1: use sampling to generate multiple documents.**

```
python mainfunc.py 
  --dataset {dataset} 
  --task step1 
  --split test 
  --num_sequence 10 
  --temperature 0.95
```

-- We note that when decoding with sample-based methods, the outputs may be different each time. So we cannot guarantee that your output will be exactly the same as the one we provide. [\[supervised: sampling\]](https://drive.google.com/drive/folders/1ZHmbodWMx1WOyyPFe60_vI6rF3piFAxg?usp=sharing)

**Method2: use sampling to generate multiple documents.**

```
python mainfunc.py 
  --dataset {dataset} 
  --task step1 
  --split test 
  --num_sequence 1 
  --temperature 0.95 
  --clustering
```

-- We note that when using different in-context demonstrations, the outputs may be different each time. So we cannot guarantee that your output will be exactly the same as the one we provide. [\[supervised: clustering\]](https://drive.google.com/drive/folders/1DNjTTOLKi24wohJKu1Z-v6b4izfymlLu?usp=sharingg)


**Fusion-in-decoder: train a reader model to infer answer from documents**

-- We use the FiD code from its official GitHub repository [\[link\]](https://github.com/facebookresearch/FiD).

**We will add the trained checkpoint to huggingface models in the next few days. If you want to get timely updates, please click "Watch".**

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
