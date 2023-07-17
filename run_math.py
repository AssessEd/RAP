import pickle
import re
from datetime import datetime
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from rap.models import QueryLlama, QueryHfModel
from rap.utils.gsm8k import judge_answer_gsm8k
from rap.math_mcts import reasoning_mcts_search

from typing import Tuple
import os
import sys
import torch
import torch.distributed
import torch.backends.cudnn
import fire
import time
import json
import random
import numpy as np
from pathlib import Path
from peft import PeftModel

from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from tqdm import tqdm
from llama import ModelArgs, Transformer, Tokenizer, LLaMA


def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    return local_rank, world_size


def load(ckpt_dir: str, tokenizer_path: str, local_rank: int, world_size: int, max_batch_size: int) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    # print(checkpoints)
    assert (
            world_size == len(checkpoints)
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(max_seq_len=2048, max_batch_size=max_batch_size, **params)
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args).cuda().half()
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)
    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator

def load_llama_hf(ckpt_dir: str, max_batch_size:int, max_response_length:int) -> LLaMA:
    from transformers import AutoTokenizer, LlamaForCausalLM
    start_time = time.time()
    # checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    # print(checkpoints, local_rank)
    # assert (
    #         world_size == len(checkpoints)
    # ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    # ckpt_path = checkpoints[local_rank]
    # print("Loading")
    # checkpoint = torch.load(ckpt_path, map_location="cpu")
    # with open(Path(ckpt_dir) / "params.json", "r") as f:
    #     params = json.loads(f.read())

    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    tokenizer = Tokenizer(AutoTokenizer.from_pretrained(ckpt_dir))    
    model = LlamaForCausalLM.from_pretrained(ckpt_dir, torch_dtype=torch.float16, device_map="auto") #.cuda().half()
    
    LORA_WEIGHTS = "/media/dev/michaelf/projects/alpaca-lora/experiments/LLaMA7B/2023-07-12-15:10:44/"
    model = PeftModel.from_pretrained(model, LORA_WEIGHTS, device_map = "auto")

    # model_args: ModelArgs = ModelArgs(max_seq_len=2048, max_batch_size=max_batch_size, **params)
    # model_args.vocab_size = tokenizer.n_words
    # tokenizer = Tokenizer(model_path=tokenizer_path)
    # torch.set_default_tensor_type(torch.cuda.HalfTensor)
    # model = Transformer(model_args).cuda().half()
    # torch.set_default_tensor_type(torch.FloatTensor)
    # model.load_state_dict(checkpoint, strict=False)
    # generator = QueryHfModel(model, tokenizer, max_batch_size=max_batch_size, max_seq_len=2048)
    generator = QueryHfModel(model, tokenizer, max_response_length=max_response_length, device="cuda:0")
    
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


def main_mcts(llama_ckpt='/media/backup/michaelf/ckpts/llamas/LLaMA-7B-HF', #'llama-ckpts/30B',
              prompts='data/gsm8k/prompts/interactive_examples.json',
              question_prompts='data/gsm8k/prompts/useful_examples.json',
              max_batch_size=2,
              max_response_length=200,
              mcts_rollouts=10,
              n_sample_subquestion=4,
              n_sample_confidence=8,
              temperature=0.8,
              max_depth=6,
              w_exp=1,
              r_alpha=0.5,
              r1_default=1,
              resume=0,
              log_dir=None,
              speedup_confidence_batch_size=2, #None
              ):
    if log_dir is None:
        log_dir = f'logs/math_mcts_{llama_ckpt.split("/")[-1]}/{datetime.now().strftime("%Y-%m%d-%H%M")}'
    os.makedirs(log_dir, exist_ok=True)

    # set random seed
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    local_rank = 0
    # local_rank, world_size = setup_model_parallel()

    # if local_rank > 0:
    #     sys.stdout = open(os.devnull, 'w')
    #     log_file = None
    # else:
    #     log_file = None

    # tokenizer_path = os.path.join(os.path.dirname(llama_ckpt), "tokenizer.model")
    # llama = load(llama_ckpt, tokenizer_path, local_rank, world_size, max_batch_size)
    # world_model = QueryLlama(llama, max_response_length=max_response_length, log_file=log_file)
    world_model = load_llama_hf(llama_ckpt, max_batch_size, max_response_length=max_response_length)  # HF - LLaMA

    examples = [
    dict(
        question="How many of the following functions are even [sin $x$ is odd and $\\cos x$ is even] (a) $f ( x )=x^{2}|x|$ (b) $f ( x )=e^{x}+e^{-x}$ (c) $f ( x )=\\log \\left[\\frac{1-x}{1+x}\\right]$ (d) $\\log \\left(\\sqrt{x^{2}+1}-x\\right)$ (e) $f ( x )=\\log \\left( x +\\sqrt{x^{2}+1}\\right.$ (f) $a^{x}-a^{-x}$ (g) $f ( x )=\\sin x+\\cos x$ (h) $\\sin x \\times\\left(e^{x}-e^{-x}\\right)$",
        answer="Here, we will first find $f(-x)$. Then we will check if $f(x)=f(-x)$ for even functions and $f(x)=-f(-x)$ for odd functions.\n(1) $\\left.f (- x )=(-x)^{2}|- x |\\right)=x^{2}|x|=f(x)$\n$\\Rightarrow$ Even function\n(2) $f (- x )=e^{-x}+e^{-(-x)}=e^{-x}+e^{x}=f(x)$\n$\\Rightarrow$ Even\n(3)\n$$\n\\begin{aligned}\nf(-x) & =\\log \\left(\\frac{1-(-x)}{1-x}\\right)=\\log \\left(\\frac{1+x}{1-x}\\right. \\\\\n& =-\\log \\left(\\frac{1-x}{1+x}=-f(x)\\right. \\\\\n& \\Rightarrow \\text { Odd function }\n\\end{aligned}\n$$\n(4)\n$$\n\\begin{array}{l}\n\\text { 4) } \\begin{aligned}\nf (- x ) & =\\log \\left(\\sqrt{\\left(-x^{2}\\right)+1-(-x)}\\right) \\\\\n= & \\log \\left(\\sqrt{\\left(-x^{2}\\right)+1}+ x \\right) \\\\\n& =-\\log \\frac{1}{\\left(\\sqrt{x^{2}+1}+x\\right)}=-\\log \\left(\\frac{\\sqrt{\\left(x^{2}\\right)+1}-x}{\\sqrt{\\left(x^{2}\\right)+1}+x \\sqrt{\\left(x^{2}\\right)+1}-x}\\right) \\\\\n& =-\\log \\left(\\frac{\\sqrt{\\left(x^{2}\\right)+1}-x}{\\sqrt{\\left(x^{2}\\right)+1}-x^{2}}\\right)=-\\log \\sqrt{\\left(x^{2}\\right)+1}- x\n\\end{aligned} \\\\\n\\Rightarrow( x )=- f ( x ) \\\\\n\\Rightarrow \\text { Odd function }\n\\end{array}\n$$\n$$\nf(x)=-f(x)\n$$\n$\\Rightarrow$ Odd function\n(5) $f (- x )=\\log \\left(\\sqrt{\\left(x^{2}\\right)+1}- x \\right)$\n$$\n\\begin{array}{l}\n=\\log \\frac{\\sqrt{\\left(x^{2}\\right)+1}-x}{\\left(\\sqrt{\\left(x^{2}\\right)+1}+x\\right)} \\times\\left(\\sqrt{\\left(x^{2}\\right)+1}+x\\right) \\\\\n=\\log \\frac{x^{2}+1-x^{2}}{\\sqrt{\\left(x^{2}\\right)+1}+x} \\\\\n=\\log \\frac{1}{\\left(\\sqrt{\\left(x^{2}\\right)+1}+x\\right)} \\\\\n=-\\log \\left(\\sqrt{\\left(x^{2}\\right)+1}+x\\right)\n\\end{array}\n$$#### 5"
    ),dict(
        question="If $f(x)=4 x-x^{2}, x \\in R$, then $f(a+1)-f(a-1)$ is equal to A. $2(4-a)$ B. $4(2-a)$ C. $4(2+a)$ D. $2(4+a)$",
        answer="The correct option is B $4(2-a)$\n$$\n\\begin{array}{l}\nf(x)=4 x-x^{2} \\\\\n\\therefore f(a+1)-f(a-1) \\\\\n=4(a+1)-(a+1)^{2}-\\left[4(a-1)-(a-1)^{2}\\right] \\\\\n=8-\\left[(a+1)^{2}-(a-1)^{2}\\right] \\\\\n=8-4 a=4(2-a)\n\\end{array}\n$$#### B"
    ),dict(
    question="Which of the following correspondences can be called a function? A. $f:\\{-1,0,1\\} \\rightarrow\\{0,1,2,3\\}$ defined by $f(x)=x^{3}$ B. $f:\\{0,1,4\\} \\rightarrow\\{-2,-1,0,1,2\\}$ defined by $f(x)= \\pm \\sqrt{x}$ C. $f:\\{0,1,4\\} \\rightarrow\\{-2,-1,0,1,2\\}$ defined by $f(x)=\\sqrt{x}$ D. $f:\\{0,1,4\\} \\rightarrow\\{-2,-1,0,1,2\\}$ defined by $f(x)=-\\sqrt{x}$",
    answer="The correct options are\nC $f:\\{0,1,4\\} \\rightarrow\\{-2,-1,0,1,2\\}$ defined by $f(x)=\\sqrt{x}$\nD $f:\\{0,1,4\\} \\rightarrow\\{-2,-1,0,1,2\\}$ defined by $f(x)=-\\sqrt{x}$\nFor option $1$ :\n$f(-1)=-1 \\notin\\{0,1,2,3\\}$\nImage of an element in domain must belong to co-domain.\nSo, $f(x)=x^{3}$ is not a function.\nFor option $2$ :\n$f(4)=-2,2$\nwhich violates the definition of the function as each element in domain should have unique image in co-domain.\nSo, $f(x)= \\pm \\sqrt{x}$ is not a function.\noption $3$ and option $4$ are functions as they satisfy all the conditions of a function.#### C and D"           
    ),dict(
    question="A mapping is selected at random from the set of all the mappings of the set $A =\\{1,2, \\ldots \\ldots ., n \\}$ into itself. The probability that the mapping selected is an injection is: A. $\\frac{1}{n^{n}}$ B. $\\frac{1}{n !}$ C. $\\frac{(n-1) !}{n^{n-1}}$ D. $\\frac{n !}{n^{n-1}}$",
    answer="The correct option is C $\\frac{(n-1) !}{n^{n-1}}$\nTotal number of functions (mappings) from $A \\rightarrow A=n^{n}$\nNumber of injective (one - one) functions\nFrom $A \\rightarrow A=n$ !\n$\\therefore$ Probability $=\\frac{n !}{n^{n}}=\\frac{(n-1) !}{n^{n-1}}$#### C $\\frac{(n-1) !}{n^{n-1}}$ #### C"       
    )]
    with open(prompts) as f:
        prompts = json.load(f)
    with open(question_prompts) as f:
        question_prompts = json.load(f)

    total_correct = [0] * mcts_rollouts
    for i, example in enumerate((pbar := tqdm(examples, disable=local_rank > 0, position=1))):
        if i < resume:
            continue
        question = example['question']
        answer = example['answer']
        answer = re.search('#### .*?([ $.0-9,\\-]+)', answer)
        answer = '' if answer is None else answer[1].replace(',', '').replace(' ', '').replace('$', '')
        trajs, tree, trees = reasoning_mcts_search(question, prompts, question_prompts, world_model,
                                                   n_sample_subquestion=n_sample_subquestion,
                                                   mcts_rollouts=mcts_rollouts,
                                                   n_sample_confidence=n_sample_confidence,
                                                   temperature=temperature,
                                                   max_depth=max_depth,
                                                   w_exp=w_exp,
                                                   r_alpha=r_alpha,
                                                   r1_default=r1_default,
                                                   eos_token_id=world_model.tokenizer.encode('\n', bos=False, eos=False)[-1],
                                                   speedup_confidence_batch_size=speedup_confidence_batch_size)
        if local_rank == 0:
            json_logs = []
            for rollout, traj in enumerate(trajs):
                output, correct = judge_answer_gsm8k(traj, answer)
                json_logs.append({
                    'rollout': rollout + 1,
                    'question': question,
                    'answer': answer,
                    'output': output,
                    'correct': correct,
                    'traj': traj,
                })
                total_correct[rollout] += correct
            with open(os.path.join(log_dir, f'{i:04d}.json'), 'w') as f:
                json.dump(json_logs, f, indent=2)
            with open(os.path.join(log_dir, f'{i:04d}.tree'), 'w') as f:
                f.write(tree)
            with open(os.path.join(log_dir, f'{i:04d}.pkl'), 'wb') as f:
                pickle.dump(trees, f)
            tqdm.write(' '.join(f'{c/(i+1-resume):0.3f}' for c in total_correct))
            pbar.set_description(f'{total_correct[-1]}/{i+1-resume}={total_correct[-1]/(i+1-resume):.2f}')


if __name__ == '__main__':
    fire.Fire(main_mcts)
