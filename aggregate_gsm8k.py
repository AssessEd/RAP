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
        reward = cur.reward
        if type(reward) != float and type(reward) != int:
            reward = reward.cpu().numpy()
        if not cur._visited or reward < 0:
            return []
        if cur.is_terminal:
            judge = judge_answer_gsm8k(cur.prompt, answer)
            cnt = 0
            while (judge, cnt) in paths:
                cnt += 1
            paths[(judge, cnt)] = (cur.prompt[re.search("Question 5:", cur.prompt).start():], [reward])
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
        rets = returns[::-1]
        questions = prompt.split('Question 5.')
        for idx, question in enumerate(questions):
            if idx < len(rets):
                acc_rets = np.mean(rets[idx:])
                question = question + f' REWARD: {rets[idx]:.3f}' + f' RETURN TO GO: {acc_rets:.3f}'
            if idx > 0:                
                questions[idx] = 'Question 5.' + question
            else:
                questions[idx] = question
        cur_ret = np.mean(rets)
        new_paths[path] = (questions, cur_ret)
        answer_dict[path[0]].append(cur_ret)
    average_answer_dict = dict()
    for key, value in answer_dict.items():
        average_answer_dict[key] = agg_fcn(value)        
    answer_reward_list = sorted(average_answer_dict.items(), key=lambda x: x[1], reverse=True)
    (output, correct), reward = answer_reward_list[0][0], answer_reward_list[0][1]
    reward_sum = sum(x[1] for x in answer_reward_list)        
    if len(answer_dict) == 0:
        return '', False, -10, 0, 0, []    
    return output, correct, reward, reward / reward_sum, reward_sum, new_paths


aggregate_choice = dict(
    max=np.max,
    mean=np.mean,
    sum=np.sum,    
)

# def main(log_dir:str='logs/gsm8k_mcts_LLaMA-7B-HF/2023-0713-2129/'): 
def main(log_dir:str='logs/gsm8k_mcts_LLaMA-7B-HF/2023-0710-0019-10-traces/', num_examples:int=-1): 
# def main(log_dir:str='logs/gsm8k_mcts_LLaMA-13B-HF/2023-0706-1705-10-traces/'): 
    print(log_dir)
    for agg_type in ['sum', 'max', 'mean']:
        agg_fcn = aggregate_choice[agg_type]
        json_files = sorted(glob.glob(f'{log_dir}/*.json'))[:num_examples]        
        print(len(json_files))
        n_aggr_correct = 0
        for n, json_file in enumerate(pbar := tqdm(json_files)):
            pickle_file = json_file.replace('.json', '.pkl')
            with open(json_file) as j_f, open(pickle_file, 'rb') as p_f:
                json_data = json.load(j_f)
                pickle_data = pickle.load(p_f)
            json_info = json_data[-1]
            pickle_info = pickle_data[-1]
            output, correct, cum_reward, conf, reward_sum, paths = aggregate(pickle_info, json_info['answer'], agg_fcn)     
            n_aggr_correct += int(correct)
            pbar.set_description(f'{n_aggr_correct}/{n+1}={n_aggr_correct/ (n+1):.3f}')
        print(f'{agg_type} aggregate correct: {n_aggr_correct}, that is {n_aggr_correct / (n+1):.3f} percent')

if __name__ == '__main__':
    fire.Fire(main)
