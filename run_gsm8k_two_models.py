import pickle
import re
from datetime import datetime
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from rap.models import QueryLlama, QueryHfModel
from rap.utils.gsm8k import judge_answer_gsm8k, get_gsm8k_dataset
from rap.gsm8k_mcts import reasoning_mcts_search

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

    tokenizer = Tokenizer(AutoTokenizer.from_pretrained(ckpt_dir))    
    model = LlamaForCausalLM.from_pretrained(ckpt_dir, load_in_4bit = True, torch_dtype=torch.float16, device_map="auto") #.cuda().half()

    ## Pretrained 
    # MATH_WEIGHTS = "/media/dev/michaelf/projects/alpaca-lora/experiments/LLaMA7B/2023-07-12-15:10:44/"
    GSM_PRETRAINED = "/media/dev/michaelf/projects/alpaca-lora/experiments/LLaMA7B/2023-07-14-20:42:22/checkpoint-2000/"
    model = PeftModel.from_pretrained(model, GSM_PRETRAINED, load_in_4bit = True, device_map = "auto")
    model.eval()
    
    # model_args: ModelArgs = ModelArgs(max_seq_len=2048, max_batch_size=max_batch_size, **params)
    # model_args.vocab_size = tokenizer.n_words
    # tokenizer = Tokenizer(model_path=tokenizer_path)
    # torch.set_default_tensor_type(torch.cuda.HalfTensor)
    # model = Transformer(model_args).cuda().half()
    # torch.set_default_tensor_type(torch.FloatTensor)
    # model.load_state_dict(checkpoint, strict=False)
    # generator = QueryHfModel(model, tokenizer, max_batch_size=max_batch_size, max_seq_len=2048)

    base_model = LlamaForCausalLM.from_pretrained(ckpt_dir, load_in_4bit = True, torch_dtype=torch.float16, device_map="auto") #.cuda().half()
    base_model.eval()
    generator = QueryHfModel(base_model, tokenizer, max_response_length=max_response_length, device="cuda:0")
    finetuned_generator = QueryHfModel(model, tokenizer, max_response_length=max_response_length, device="cuda:0")
    
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator, finetuned_generator


def main_mcts(llama_ckpt='/media/backup/michaelf/ckpts/llamas/LLaMA-7B-HF', #'llama-ckpts/30B',
            #   prompts='data/gsm8k/prompts/finetuned_prompt_61.json',
            #   question_prompts='data/gsm8k/prompts/useful_examples_.json',
              prompts='data/gsm8k/prompts/finetuned_prompt_answers_only.json',
            #   prompts='data/gsm8k/prompts/interactive_examples.json', 
              question_prompts='data/gsm8k/prompts/useful_examples.json',              
              max_batch_size=2,
              max_response_length=200,
              mcts_rollouts=10,
              n_sample_subquestion=4,
              n_sample_confidence=8,
              temperature=.6,
              max_depth=6, 
              w_exp=1,
              r_alpha=0.5,
              r1_default=1,
              resume=0,
              finish=101,
              log_dir=None,
              node_visit_penalty=0.0,
              seed=42,
              discount=0.95,
              speedup_confidence_batch_size=2, 
              ):
    mcts_type = 'mean'
    if mcts_type == 'mean':
        mcts_args = dict(prior=True, aggr_reward='mean', aggr_child='max')
        if log_dir is None:
            log_dir = f'logs/gsm8k_mcts_{llama_ckpt.split("/")[-1]}/{datetime.now().strftime("%Y-%m%d-%H%M")}' + f' temperature={temperature}' + 'gsm_pretrained_only'
    elif mcts_type == 'accumulated':
        mcts_args = dict(prior=True, discount=discount, node_visit_penalty=node_visit_penalty, aggr_child='max')
        if log_dir is None:
            log_dir = f'logs/gsm8k_mcts_{llama_ckpt.split("/")[-1]}/{datetime.now().strftime("%Y-%m%d-%H%M")}' + f' temperature={temperature}' + f' penalty={node_visit_penalty}, '+ f'discount={discount}'
    else:
        raise NotImplementedError
    os.makedirs(log_dir, exist_ok=True)
    params = dict(
              prompts=prompts,
              seed=seed,
              llama_ckpt=llama_ckpt,
              question_prompts=question_prompts,
              max_batch_size=max_batch_size,
              max_response_length=max_response_length,
              mcts_rollouts=mcts_rollouts,
              n_sample_subquestion=n_sample_subquestion,
              n_sample_confidence=n_sample_confidence,
              temperature=temperature,
              max_depth=max_depth,
              w_exp=w_exp,
              r_alpha=r_alpha,
              r1_default=r1_default,
              resume=resume,
              finish=finish,
              log_dir=log_dir,
              speedup_confidence_batch_size=speedup_confidence_batch_size,
              mcts_type=mcts_type,
              mcts_args=mcts_args
            )
    with open(log_dir + "/params.json", "w") as f:
        json.dump(params, f, indent=2)
    # set random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
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
    world_model, finetuned_world_model = load_llama_hf(llama_ckpt, max_batch_size, max_response_length=max_response_length)  # HF - LLaMA

    examples = get_gsm8k_dataset('test')[:finish]
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
        trajs, tree, trees = reasoning_mcts_search(question, prompts, question_prompts, world_model, finetuned_world_model,
                                                   n_sample_subquestion=n_sample_subquestion,
                                                   mcts_rollouts=mcts_rollouts,
                                                   n_sample_confidence=n_sample_confidence,
                                                   temperature=temperature,
                                                   max_depth=max_depth,
                                                   w_exp=w_exp,
                                                   r_alpha=r_alpha,
                                                   r1_default=r1_default,
                                                   eos_token_id=world_model.tokenizer.encode('\n', bos=False, eos=False)[-1],
                                                   speedup_confidence_batch_size=speedup_confidence_batch_size,
                                                   mcts_type=mcts_type,
                                                   mcts_args=mcts_args)
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
