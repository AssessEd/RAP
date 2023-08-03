import io
import os
import random
import re
import warnings
from collections import defaultdict
from copy import deepcopy
from tqdm import tqdm, trange
import torch
from .mcts import AccumulatedRewardMCTS, MCTS, MCTSNode
from .models import QueryLM
import numpy as np
from transformers import StoppingCriteria, StoppingCriteriaList
from typing import List

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops = [], encounters = 1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False

#===

# tokeniser.decode([13]) returns "\n"



def is_terminal_question(prompt, prompt_index=0):
    if not prompt: 
        return True 
    prompt = prompt.split('\n\n')[-1]
    if '####' in prompt:
        return True
    return False


class ReasoningMCTSNode(MCTSNode):
    @property
    def visited(self):
        return self._visited

    def __init__(self, prompt, gen_fn, depth,
                 parent: 'ReasoningMCTSNode' = None, r0 = 0.0):
        self._conf = None
        self.children = []
        self.prompt = prompt
        # self.question_prompt = question_prompt
        self.gen_fn = gen_fn
        self.depth = depth
        self._r0 = r0
        self._ans_list = None
        self._visited = False
        self.parent = parent

    def _child_node(self, prompt, r0):
        return ReasoningMCTSNode(prompt, self.gen_fn, self.depth + 1,
                                 parent=self, r0=r0)

    def _get_children(self):
        self._visited = True
        if self.is_terminal:
            return self.children
        questions, r0s = self.gen_fn(self.prompt, self.depth)
        for question, r0 in zip(questions, r0s):
            self.children.append(self._child_node(question, r0))
        return self.children

    def find_children(self):
        self.children = self.children or self._get_children()
        return self.children

    def find_one_child(self) -> MCTSNode:
        return random.choice(self.find_children())

    def _static_terminal(self):
        return is_terminal_question(self.prompt)

    @property
    def is_terminal(self):
        return self._static_terminal() or self.reward < -1

    @property
    def reward(self):
        return self._r0 

    def print(self, mcts: MCTS, file=None, acc_reward:float=0.0, discount:float=1.0, log_acc_reward:float=0.0):
        def pprint(*args):
            if file is None:
                tqdm.write(*args)
            else:
                print(*args, file=file)
        p1 = '-' * (4 * self.depth - 4)
        prefix = ' ' * (4 * self.depth - 4)
        if not self.prompt:
            return
        depth = self.depth - 1
        generated_text = self.prompt.split(f'Problem:\n')[-1].split('\n')
        if self.depth > 1:
            question = f'Step {self.depth-1}: ' + generated_text[-2]
        elif self.depth == 1:
            question = 'Problem: ' + generated_text[-3]
            depth = 1            
        pprint(p1 + question)
        acc_reward += discount ** (self.depth - 2) * self.reward
        if self.reward > 0:
            log_acc_reward += np.log(self.reward)            
        pprint(prefix + f'R: {self.reward:.3f} ; N: {mcts.N[self]} ; M: {mcts.M[self]:.3f}; Acc. Reward: {acc_reward:.3f}; Log acc Reward: {log_acc_reward:.3f}; Mean Reward: {acc_reward/depth:.3f}') # ; r0 : {self._r0:.3f}')
        if not self.visited:
            return
        # answer = 'A' + self.prompt.split(f'Solution:')[-1].split('\n')[0]
        if self.reward < -1:
            if file is not None:
                pprint(prefix + question)
                # pprint(prefix + answer)
            return
        # if self.parent is not None:
        #     if file is not None:
        #         pprint(prefix + answer)
        #     match = re.match('.*The answer is (.*)\\.', answer)
        #     if match is not None:
        #         term = '\u25A1' if self.is_terminal else ''
        #         pprint(prefix + f'answer: {match[1]} ; ans_list: {self._ans_list} ; r1 : {self._r1:.3f}{term}')
            # if not self.is_terminal:
            #     pprint(prefix + f'conf_list : {self._conf_list} ; r2 : {self._r2:.3f}')
        for child in self.children:
            child.print(mcts, file, acc_reward, discount=discount, log_acc_reward=log_acc_reward)
        if self.depth == 1:
            pprint("=" * 12)

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.gen_fn is None or self.reward_fn is None:
            warnings.warn('MCTSNode loaded from pickle is read-only; Do not further roll out the tree!')

    def __getstate__(self):
        state = self.__dict__.copy()
        state['gen_fn'] = None
        state['reward_fn'] = None
        return state


mcts_algo = dict(
    mean=MCTS,
    accumulated=AccumulatedRewardMCTS    
)

def adjust_prompt(prompt, tokenizer, prompts):
    prompt_shape = tokenizer.sp_model(prompt, return_tensors="pt")['input_ids'].shape
    while prompt_shape[1] >= 1950:
        prompt_list = prompt.split(prompts["question_prefix"])
        if prompt_list:
            if not prompt_list[0]:
                prompt = prompts["question_prefix"] + prompts["question_prefix"].join(prompt_list[2:])
            else:
                prompt = prompts["question_prefix"] + prompts["question_prefix"].join(prompt_list[1:])
            prompt_shape = tokenizer.sp_model(prompt, return_tensors="pt")['input_ids'].shape
        else:
            break
    return prompt 

def reasoning_mcts_search(question: str,
                          prompts,
                          demonstrations:List[str],
                          world_model: QueryLM,
                          finetuned_world_model: QueryLM,
                          n_sample_subquestion,
                          temperature,
                          mcts_rollouts,
                          w_exp,
                          n_sample_confidence,
                          max_depth,
                          eos_token_id,
                          speedup_confidence_batch_size=None,
                          mcts_type='mean',
                          mcts_args=dict()):
    if speedup_confidence_batch_size is None:
        speedup_confidence_batch_size = n_sample_confidence
    stop_word_id = [torch.tensor([13, 13]).to(finetuned_world_model.device)]
    criterion = StoppingCriteriaList([StoppingCriteriaSub(stops = stop_word_id)])
    if demonstrations:
        input_prompts = "\n\n".join([prompts["question_prefix"] + demo['question'] + "\n" + prompts['solution_prefix']  + demo['answer'] for demo in demonstrations]) + "\n\n"
    else:
        input_prompts = prompts["input"]
    parsed_question = (prompts["question_prefix"] + question  + prompts['solution_prefix']).replace("$", "\$")
    base_generation_prompt = input_prompts.replace("$", "\$") + parsed_question
    base_evaluation_prompt = prompts["evaluation_input"].replace("$", "\$") + parsed_question

    def gen_fn(agent_input, depth):
        if depth == max_depth:
            agent_input += "####"
            num_return_sequences = 1
        else:
            num_return_sequences = n_sample_subquestion
        adjusted_agent_input = adjust_prompt(agent_input + "", finetuned_world_model.tokenizer, prompts)
        agent_output = finetuned_world_model.query_LM(
            adjusted_agent_input,
            eos_token_id=eos_token_id
            )
        if num_return_sequences > 1:
            agent_output += finetuned_world_model.query_LM(
                adjusted_agent_input, 
                do_sample=True, 
                num_return_sequences=num_return_sequences-1,
                eos_token_id=eos_token_id, 
                temperature=temperature
            )
        for i, output in enumerate(agent_output):
            if is_terminal_question(output):    
                out_list = output.split('\n\n')
                res = re.findall('#### .*?([$.0-9,\\-]+)', out_list[-1])
                if res and re.search('#### ' + res[0], out_list[-1]): 
                    out_list[-1] = out_list[-1][:re.search('#### ' + res[0], out_list[-1]).end()].replace(",", "")
                    output_ = "\n\n".join(out_list) + "\n"
                else:
                    print(res)
                    output_ = output 
                agent_output[i] = output_

        # unique the output
        # set does not guarantee order ; dict guarantees insertion order
        agent_output = list(dict.fromkeys(agent_output))
        partial_solutions = [o.split(prompts["solution_prefix"])[-1] for o in agent_output]
        
        r0 = [r0_fn(inp) for inp in partial_solutions]
        return agent_output, r0

    def r0_fn(inp):
        inp = adjust_prompt(base_evaluation_prompt + inp, world_model.tokenizer, prompts)
        agent_output = world_model.query_LM_answers(
            prompts["evaluation_instructions"] + inp + prompts['evaluation_postfix'],
            return_dict_in_generate = True,
            output_scores = True,
            top_k = 1,
            stopping_criteria = criterion
        )
        text = world_model.tokenizer.sp_model.batch_decode(agent_output.sequences, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        scores = agent_output.scores
        score_re = re.search(prompts['evaluation_postfix'], text)  
        if score_re:
            scores_ = scores[0].cpu()[:,world_model.yes_no]
            dist = torch.softmax(scores_.float(), dim=1)                
            score = dist[:, 0].cpu().numpy()[0]
        else:
            score = 0.0
        return score


    # input_prompts = prompts["input"] + prompts["question_prefix"] + question.strip() + "\nSolution:"
    # input_question_prompts = question_prompts["input"] + question_prompts["question_prefix"] + question.strip() + "\nSolution:"
    mcts_args['w_exp'] = w_exp
    mcts_class = mcts_algo[mcts_type]
    mcts = mcts_class(**mcts_args)
    root = ReasoningMCTSNode(base_generation_prompt, gen_fn,
                             depth=1)
    trajs = []
    trees = []
    for _ in (pbar := trange(mcts_rollouts, disable=bool(int(os.environ.get("LOCAL_RANK", -1))), position=0)):
        mcts.rollout(root)
        root.print(mcts)
        max_n, max_r = mcts.max_mean_terminal(root) # maximum average reward per path
        trajs.append(traj := max_n.prompt.split('\n\n')[-1])
        output = re.findall('#### (.+).*', traj)
        if len(output) == 0:
            temp_r = 'not found'
        else:
            temp_r = output[-1].replace(',', '')
        pbar.set_description(f'{max_r:.3f} {temp_r}')
        tree_copy = deepcopy(root)
        tree_copy.Q = dict(mcts.Q)
        tree_copy.N = dict(mcts.N)
        tree_copy.M = dict(mcts.M)
        trees.append(tree_copy)

    with io.StringIO() as f:
        root.print(mcts, file=f)
        tree = f.getvalue()
    return trajs, tree, trees
