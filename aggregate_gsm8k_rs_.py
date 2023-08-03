import glob
import json
import pickle
from collections import defaultdict
import re
import fire
from tqdm import tqdm
import numpy as np
from rap.gsm8k_mcts import ReasoningMCTSNode
from rap.utils.gsm8k import judge_answer_gsm8k_pretrained
import os 
import torch
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

def aggregate(root: ReasoningMCTSNode, answer, agg_fcn, params):
    answer_dict = defaultdict(list)
    paths = defaultdict()
    penalty = params.get('penalty', 0)
    discount = params.get('discount', 1)
    reward_type = params.get('mcts_type', 'mean')
    num_demonstrations = params.get('num_demonstrations', -1)
    def visit(cur: ReasoningMCTSNode):
        reward = cur.reward
        # if type(reward) != float and type(reward) != int:
        #     reward = reward.cpu().numpy()
        if not cur._visited or reward < 0:
            return []
        if cur.is_terminal:
            # print(cur.prompt)
            judge = judge_answer_gsm8k_pretrained(cur.prompt, answer)
            cnt = 0
            while (judge, cnt) in paths:
                cnt += 1
            try:     
                cur_prompt = cur.prompt.split('\n\n')[num_demonstrations]
            except IndexError:
                cur_prompt = cur.prompt.split('\n\n')[-1]
            # except AttributeError:
            #     cur_prompt = cur.prompt 
            #     print(cur_prompt)
            paths[(judge, cnt)] = (cur_prompt, [reward])
            return [(judge, cnt)]
        cur_list = []
        for child in cur.children:
            cur_list.extend(child_info := visit(child))
        for judge_cnt in cur_list:
            paths[judge_cnt] = (paths[judge_cnt][0], paths[judge_cnt][1] + [reward])            
        return cur_list

    visit(root)

    new_paths = dict()
    for path, (prompt, returns) in paths.items():
        # print(path, prompt)
        rets = returns[::-1]
        # if "Solution:" in prompt:
        #     prompt = prompt.split("Solution:")[-1]        
        #     steps = prompt.split('\n')
        #     for idx, step in enumerate(steps):
        #         if idx < len(rets):
        #             acc_rets = np.mean(rets[idx:])
        #             step = step + f' REWARD: {rets[idx]:.3f}' + f' RETURN TO GO: {acc_rets:.3f}'
        #         if idx > 0:                
        #             steps[idx] = 'Problem:' + step
        #         else:
        #             steps[idx] = step
        if reward_type == 'mean':
            cur_ret = np.mean(rets)
        elif reward_type == 'accumulated':   
            cur_ret = 0        
            for reward in returns:        
                cur_ret = discount * cur_ret + reward - penalty
        # new_paths[path] = (questions, cur_ret)
        answer_dict[path[0]].append(cur_ret)
    average_answer_dict = dict()
    for key, value in answer_dict.items():
        average_answer_dict[key] = agg_fcn(value)  
    # print(average_answer_dict)      
    answer_reward_list = sorted(average_answer_dict.items(), key=lambda x: x[1], reverse=True)
    if len(answer_reward_list) == 0:
        return '', False, -10, 0, 0, [] 
    (output, correct), reward = answer_reward_list[0][0], answer_reward_list[0][1]
    reward_sum = sum(x[1] for x in answer_reward_list)        
   
    return output, correct, reward, reward / reward_sum, reward_sum, new_paths


aggregate_choice = dict(
    max=np.max,
    mean=np.mean,
    sum=np.sum,    
)

# def main(log_dir:str='logs/gsm8k_mcts_LLaMA-7B-HF/2023-0713-2129/'): 
def main(log_dir:str='logs/test/gsm8k_mcts_LLaMA-7B-HF/2023-0731-2103 temperature=0.3gsm_pretrained_only', num_examples:int=-1): 
# def main(log_dir:str='logs/gsm8k_mcts_LLaMA-13B-HF/2023-0706-1705-10-traces/'): 
    print(log_dir)
    for agg_type in ['sum']: #, 'max', 'mean']:
        agg_fcn = aggregate_choice[agg_type]
        all_files = set(glob.glob(f'{log_dir}/*.json'))
        params = dict()
        if f"{log_dir}/params.json" in all_files:
            all_files.remove(f"{log_dir}/params.json")
            with open(f"{log_dir}/params.json") as f:
                params = json.load(f)
        if f"{log_dir}/params_.json" in all_files:
            all_files.remove(f"{log_dir}/params_.json")
        json_files = sorted(all_files)[:num_examples]        
        # print(len(json_files))
        n_aggr_correct = 0
        correct_problems = []
        for n, json_file in enumerate(pbar := tqdm(json_files)):
            pickle_file = json_file.replace('.json', '.pkl')
            with open(json_file) as j_f, open(pickle_file, 'rb') as p_f:
                json_data = json.load(j_f)
                pickle_data = pickle.load(p_f)
            json_info = json_data[-1]
            pickle_info = pickle_data[-1]
            output, correct, cum_reward, conf, reward_sum, paths = aggregate(pickle_info, json_info['answer'], agg_fcn, params)     
            n_aggr_correct += int(correct)
            pbar.set_description(f'{n_aggr_correct}/{n+1}={n_aggr_correct/ (n+1):.3f}')
            if correct:
                correct_problems.append(n)
        print(f'{agg_type} aggregate correct: {n_aggr_correct}, that is {n_aggr_correct / (n+1):.3f} percent')
        print(correct_problems)
if __name__ == '__main__':
    fire.Fire(main)
