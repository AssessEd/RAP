import unittest

import glob
import json
import pickle
from collections import defaultdict

import fire
from tqdm import tqdm
import warnings
import numpy as np
from rap.gsm8k_mcts import ReasoningMCTSNode
from rap.utils.gsm8k import judge_answer_gsm8k
from aggregate_gsm8k import aggregate as aggregate_new
import re

def aggregate(root: ReasoningMCTSNode, answer):
    answer_dict = defaultdict(lambda: 0)

    def visit(cur: ReasoningMCTSNode):
        reward = cur.reward
        if type(reward) != float:
            reward = reward.cpu().numpy()
        if not cur._visited or reward < 0:
            return []

        if cur.is_terminal:
            judge = judge_answer_gsm8k(cur.prompt, answer)
            answer_dict[judge] += reward / cur.depth
            return [(judge, cur.depth)]
        cur_depth_list = defaultdict(list)
        cur_list = []
        for child in cur.children:
            cur_list.extend(child_info := visit(child))
            for judge, depth in child_info:
                cur_depth_list[judge].append(depth)
        for judge, depths in cur_depth_list.items():
            answer_dict[judge] += reward * len(depths) / sum(depths)
        return cur_list

    visit(root)

    if len(answer_dict) == 0:
        return '', False, -10, 0

    answer_reward_list = sorted(answer_dict.items(), key=lambda x: x[1], reverse=True)
    (output, correct), reward = answer_reward_list[0]
    reward_sum = sum(x[1] for x in answer_reward_list)
    return output, correct, reward, reward / reward_sum

def aggregate_by_path(root: ReasoningMCTSNode, answer, agg_fcn):
    answer_dict = defaultdict(lambda:0)
    paths = defaultdict()

    def visit(cur: ReasoningMCTSNode):
        reward = cur.reward
        if type(reward) != float:
            reward = reward.cpu().numpy()
        if not cur._visited or reward < 0:
            return []
        if cur.is_terminal:
            judge = judge_answer_gsm8k(cur.prompt, answer)
            cnt = 0
            while (judge, cnt) in paths:
                cnt += 1
            paths[(judge, cnt)] = (cur.prompt[re.search("Question 5:", cur.prompt).start():], [reward + 0.0])
            answer_dict[judge] += reward / cur.depth
            return [(judge, cnt, cur.depth)]
        cur_depth_list = defaultdict(list)
        cur_list = []
        for child in cur.children:
            cur_list.extend(child_info := visit(child))
            for judge, cnt, depth in child_info:
                cur_depth_list[(judge, cnt)].append(depth)
        for judge_cnt, depths in cur_depth_list.items():
            answer_dict[judge_cnt[0]] += reward  * len(depths) / sum(depths)
            paths[judge_cnt] = (paths[judge_cnt][0], paths[judge_cnt][1] + [reward + 0.0])            
        return cur_list

    visit(root)

    average_answer_dict = dict()
    for key, value in answer_dict.items():
        average_answer_dict[key] = agg_fcn(value)             
    answer_reward_list = sorted(average_answer_dict.items(), key=lambda x: x[1], reverse=True)
    (output, correct), reward = answer_reward_list[0][0], answer_reward_list[0][1]
    reward_sum = sum(x[1] for x in answer_reward_list)        
    if len(answer_dict) == 0:
        return '', False, -10, 0, 0, []    
    return output, correct, reward, reward / reward_sum, reward_sum

class TestAggregate(unittest.TestCase):
    def test_aggregate_fcn(self):
        log_dir='./tests/test_files/'
        json_files = sorted(glob.glob(f'{log_dir}/*.json'))
        correct_cnt: dict[int, list[bool]] = defaultdict(list)
        confs: dict[int, list[bool]] = defaultdict(list)
        cum_rewards: dict[int, list[bool]] = defaultdict(list)
        correct_cnt_new: dict[int, list[bool]] = defaultdict(list)
        confs_new: dict[int, list[bool]] = defaultdict(list)
        cum_rewards_new: dict[int, list[bool]] = defaultdict(list)
        for j, json_file in enumerate(pbar := tqdm(json_files)):
            pickle_file = json_file.replace('.json', '.pkl')
            with open(json_file) as j_f, open(pickle_file, 'rb') as p_f:
                json_data = json.load(j_f)
                pickle_data = pickle.load(p_f)
            cur_cum_rewards = []
            for i, (json_info, pickle_info) in enumerate(zip(json_data, pickle_data)):
                _, correct, cum_reward, conf, _ = aggregate_by_path(pickle_info, json_info['answer'], agg_fcn=np.sum)
                correct_cnt[i].append(correct)
                confs[i].append(conf)
                cum_rewards[i].append(cum_reward)
                cur_cum_rewards.append(cum_reward)
                _, correct_new, cum_reward_new, conf_new, _, _ = aggregate_new(pickle_info, json_info['answer'], agg_fcn=np.sum)
                correct_cnt_new[i].append(correct_new)
                confs_new[i].append(cum_reward_new)
                cum_rewards_new[i].append(conf_new)
                # print(cum_reward_new, cum_reward, correct, correct_new)
                assert correct_new == correct, f"correctness don't match in {i}-th run and {j}-th file"
                assert np.allclose(cum_reward_new, cum_reward), f"returns don't match in {i}-th run and {j}-th file"
                assert np.allclose(conf_new, conf), f"returns don't match in {i}-th run and {j}-th file"
            v = correct_cnt[i]
            pbar.set_description(f'{sum(v)}/{len(v)}={sum(v) / len(v):.3f}')
            # print(cur_cum_rewards)
            assert np.allclose(cur_cum_rewards[np.argmax(cur_cum_rewards)], cur_cum_rewards[-1]), "the best return is not last"
        for k, v in correct_cnt.items():
            print(f'{k}: {sum(v)}/{len(v)}={sum(v) / len(v)}')


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=ImportWarning)
        unittest.main(warnings='ignore')