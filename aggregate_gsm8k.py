import glob
import json
import pickle
from collections import defaultdict

import fire
from tqdm import tqdm
import numpy as np
from rap.gsm8k_mcts import ReasoningMCTSNode
from rap.utils.gsm8k import judge_answer_gsm8k
import os 
import torch
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

def aggregate(root: ReasoningMCTSNode, answer):
    answer_dict = defaultdict(lambda: 0)

    def visit(cur: ReasoningMCTSNode):
        if not cur._visited or cur.reward < 0:
            return []
        if cur.is_terminal:
            judge = judge_answer_gsm8k(cur.prompt, answer)
            answer_dict[judge] += cur.reward / cur.depth
            return [(judge, cur.depth)]
        cur_depth_list = defaultdict(list)
        cur_list = []
        for child in cur.children:
            cur_list.extend(child_info := visit(child))
            for judge, depth in child_info:
                cur_depth_list[judge].append(depth)
        for judge, depths in cur_depth_list.items():
            answer_dict[judge] += cur.reward * len(depths) / sum(depths)
        return cur_list

    visit(root)

    if len(answer_dict) == 0:
        return '', False, -10, 0, 0

    answer_reward_list = sorted(answer_dict.items(), key=lambda x: x[1], reverse=True)
    (output, correct), reward = answer_reward_list[0][0], answer_reward_list[0][1].cpu().numpy()
    reward_sum = sum(x[1] for x in answer_reward_list).cpu().numpy()
    return output, correct, reward, reward / reward_sum, reward_sum


# def main(log_dir:str='logs/gsm8k_mcts_LLaMA-7B-HF/2023-0710-0019/'): #'logs/gsm8k_mcts_LLaMA-13B-HF/2023-0706-1705/'
def main(log_dir:str='logs/gsm8k_mcts_LLaMA-13B-HF/2023-0706-1705/'): #'logs/gsm8k_mcts_LLaMA-13B-HF/2023-0706-1705/'
    json_files = sorted(glob.glob(f'{log_dir}/*.json'))#[367:]
    correct_cnt: dict[int, list[bool]] = defaultdict(list)
    cum_rewards: dict[int, list[float]] = defaultdict(list)
    n_correct = n_aggr_correct = 0
    for json_file in (pbar := tqdm(json_files)):
        pickle_file = json_file.replace('.json', '.pkl')
        with open(json_file) as j_f, open(pickle_file, 'rb') as p_f:
            json_data = json.load(j_f)
            pickle_data = pickle.load(p_f)
        is_correct = False
        rewards = []
        reward_sums = []
        confs = []
        for i, (json_info, pickle_info) in enumerate(zip(json_data, pickle_data)):            
            output, correct, cum_reward, conf, reward_sum = aggregate(pickle_info, json_info['answer'])
            is_correct = is_correct or json_info['correct']
            correct_cnt[i].append(correct)
            rewards.append(cum_reward)
            confs.append(conf)
            reward_sums.append(reward_sum)
        n_correct += int(is_correct)
        aggr_correct = int(correct_cnt[np.stack(rewards).argmax()][-1])
        n_aggr_correct += aggr_correct
        if int(is_correct) != aggr_correct:
            print(json_file, np.stack(rewards).argmax(), is_correct, correct_cnt)
        v = correct_cnt[i]
        pbar.set_description(f'{sum(v)}/{len(v)}={sum(v) / len(v):.3f}')
    for k, v in correct_cnt.items():
        print(f'{k}: {sum(v)}/{len(v)}={sum(v) / len(v)}')
    print(f'At least one correct: {n_correct}, that is {n_correct / len(v):.3f} percent')
    print(f'Aggregate correct: {n_aggr_correct}, that is {n_aggr_correct / len(v):.3f} percent')

if __name__ == '__main__':
    fire.Fire(main)
