import glob
import json
import pickle
from collections import defaultdict
import re
import fire
from tqdm import tqdm
import numpy as np
from rap.gsm8k_mcts import ReasoningMCTSNode
from rap.utils.gsm8k import judge_answer_gsm8k
import os 
import torch
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

def aggregate(root: ReasoningMCTSNode, answer, agg_fcn):
    answer_dict = defaultdict(list)
    paths = defaultdict()

    def visit(cur: ReasoningMCTSNode):
        if not cur._visited or cur.reward < 0:
            return []
        if cur.is_terminal:
            judge = judge_answer_gsm8k(cur.prompt, answer)
            answer_dict[judge].append(cur.reward / cur.depth)
            cnt = 0
            while (judge, cnt) in paths:
                cnt += 1
            paths[(judge, cnt)] = (cur.prompt[re.search("Question 5:", cur.prompt).start():], [cur.reward])
            return [(judge, cnt, cur.depth)]
        cur_depth_list = defaultdict(list)
        cur_list = []
        for child in cur.children:
            cur_list.extend(child_info := visit(child))
            for judge, cnt, depth in child_info:
                cur_depth_list[(judge, cnt)].append(depth)
        for judge_cnt, depths in cur_depth_list.items():
            answer_dict[judge_cnt[0]][-1] += cur.reward  * len(depths) / sum(depths)
            paths[judge_cnt] = (paths[judge_cnt][0], paths[judge_cnt][1] + [cur.reward])            
        return cur_list

    visit(root)

    if len(answer_dict) == 0:
        return '', False, -10, 0, 0, []
    average_answer_dict = dict()
    for key, value in answer_dict.items():
        average_answer_dict[key] = agg_fcn(torch.tensor(value))
    answer_reward_list = sorted(average_answer_dict.items(), key=lambda x: x[1], reverse=True)
    (output, correct), reward = answer_reward_list[0][0], answer_reward_list[0][1].cpu().numpy()
    reward_sum = sum(x[1] for x in answer_reward_list).cpu().numpy()
    for path, (prompt, returns) in paths.items():
        rets = torch.tensor(returns).cpu().numpy()[::-1]
        questions = prompt.split('Question 5.')
        for idx, question in enumerate(questions):
            if idx < len(rets):
                acc_rets = np.mean(rets[idx:])
                question = question + f' REWARD: {rets[idx]:.3f}' + f' RETURN TO GO: {acc_rets:.3f}'
            if idx > 0:                
                questions[idx] = 'Question 5.' + question
            else:
                questions[idx] = question
        paths[path] = (questions, list(rets), np.mean(rets))
    return output, correct, reward, reward / reward_sum, reward_sum, paths

aggregate_choice = dict(
    max=torch.sum,
    mean=torch.mean,
    sum=torch.max,    
)

def main(log_dir:str='logs/gsm8k_mcts_LLaMA-7B-HF/2023-0710-0019/', agg_type='mean'): 
# def main(log_dir:str='logs/gsm8k_mcts_LLaMA-13B-HF/2023-0706-1705/', agg_type='sum'): 
    for agg_type in ['max', 'mean', 'sum']:
        agg_fcn = aggregate_choice[agg_type]
        json_files = sorted(glob.glob(f'{log_dir}/*.json'))
        print(len(json_files))
        correct_cnt: dict[int, list[bool]] = defaultdict(list)
        cum_rewards: dict[int, list[float]] = defaultdict(list)
        n_correct = n_aggr_correct = n_last_correct = 0
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
                output, correct, cum_reward, conf, reward_sum, paths = aggregate(pickle_info, json_info['answer'], agg_fcn)
                is_correct = is_correct or json_info['correct']
                # last_correct = json_info['correct']
                correct_cnt[i].append(correct)
                rewards.append(cum_reward)
                confs.append(conf)
                reward_sums.append(reward_sum)
            n_correct += int(is_correct)
            n_last_correct += int(json_info['correct'])
            aggr_correct = int(correct_cnt[np.stack(rewards).argmax()][-1])
            n_aggr_correct += aggr_correct
            # if int(is_correct) != aggr_correct:
            #     print(json_file, np.stack(rewards).argmax(), is_correct, correct_cnt)
            v = correct_cnt[i]
            pbar.set_description(f'{sum(v)}/{len(v)}={sum(v) / len(v):.3f}')
        # for k, v in correct_cnt.items():
        #     print(f'{k}: {sum(v)}/{len(v)}={sum(v) / len(v)}')
        print(f'At least one is correct: {n_correct}, that is {n_correct / len(v):.3f} percent')
        print(f'The last one is correct: {n_last_correct}, that is {n_last_correct / len(v):.3f} percent')
        print(f'Aggregate is correct: {n_aggr_correct}, that is {n_aggr_correct / len(v):.3f} percent')

if __name__ == '__main__':
    fire.Fire(main)
