import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# import pandas as pd
import numpy as np
import torch
import glob
import json
import re
from rap.utils.gsm8k import get_gsm8k_dataset
from transformers import LlamaTokenizer, LlamaForCausalLM
# from IPython.display import display, HTML, Latex
# from utils.callbacks import Iteratorize, Stream
# from sympy.parsing.latex import parse_latex
# from utils.prompter import Prompter
from collections import Counter
import datetime
from pathlib import Path
# from peft import PeftModel

# PATH_TO_DATA = './data/gsm8k'
BASE_MODEL = '/media/backup/michaelf/ckpts/llamas/LLaMA-13B-HF'
current_time = datetime.datetime.now()
time_stamp = "-".join("".join(current_time.__str__()[:16].split(":")).split(" "))
LOG_PATH = Path('logs/gsm8k_SC/') / Path(time_stamp)
os.makedirs(LOG_PATH)
# files = sorted(glob.glob(PATH_TO_DATA + "test.jsonl"))
examples = get_gsm8k_dataset('test')
# indices = [1,2,3,4,5,6,7,8,9,10] # [9, 42, 57, 64, 250] #, 300, 325, 666, 700, 701] # 10 shots.

with open("data/gsm8k/prompts/prompt_sc.json") as f:
    BASE_PROMPT = json.load(f)["input"]
    
# problems = []
# solutions = []

# for index in indices:
    
#     print(index)
    
#     # with open(files[index]) as f:
#     #     data = json.load(f)
#     data = examples[index]     
#     problem, solution = data["question"], data["answer"]
    
#     problems.append(problem)
#     solutions.append(solution)
    

# briefs = []

# brief1 = "Final answer: The final answer is {}".format("$x=\\boxed{-\\frac{1}{8}}$")
# brief2 = "Final answer: The final answer is {}".format("$\\boxed{2}$")
# brief3 = "Final answer: The final answer is {}".format("$\\boxed{\\frac{2}{5}}$")
# brief4 = "Final answer: The final answer is {}".format("$\\boxed{4}$")
# brief5 = "Final answer: The final answer is {}".format("$\\boxed{3}$")
# brief6 = "Final answer: The final answer is {}".format("$\\boxed{28}$")
# brief7 = "Final answer: The final answer is {}".format("$\\boxed{7}$")
# brief8 = "Final answer: The final answer is {}".format("$\\boxed{10}$")
# brief9 = "Final answer: The final answer is {}".format("$\\boxed{137 \\frac{1}{2}}$")
# brief10 = "Final answer: The final answer is {}".format("$\\boxed{4}$")

# briefs.append(brief1)
# briefs.append(brief2)
# briefs.append(brief3)
# briefs.append(brief4)
# briefs.append(brief5)
# briefs.append(brief6)
# briefs.append(brief7)
# briefs.append(brief8)
# briefs.append(brief9)
# briefs.append(brief10)

# BASE_PROMPT = ""

tokeniser = LlamaTokenizer.from_pretrained(BASE_MODEL)

#=== Only need this.

tokeniser.pad_token_id = 0
# eos_token_id = tokeniser.encode('\n')[-1]

model = LlamaForCausalLM.from_pretrained(BASE_MODEL, #load_in_8bit = True, 
                                        torch_dtype = torch.float16,
                                        device_map = "auto")

#===

model.eval();
NUM_SAMPLES = 50 # From LLaMA paper, we have 256 for MATH.

TEMP = 0.8 # T = 0 is the same as greedy decoding. T = 0.1 for 13B.
TOP_K = 20
MAX_NEW_TOKENS = 256 #128 for 13B.

# for problem, solution in zip(problems, solutions):
    
#     PROBLEM = """
#               Question:
#               {}
#               Answer:
#               {}              
#               """.format(problem, solution)
    
#     BASE_PROMPT = BASE_PROMPT + PROBLEM
#     #break
# BASE_PROMPT = 
BASE_PROMPT +=  "Question 5: " 

for test_index  in range(17,20):# len(examples)):        #1004 = WRONG #1003 = WRONG #1002 = WRONG #1001 = WRONG ONLY LAST STEP (1 SIGN) #1000 = RIGHT
    # if test_index in indices:
        # continue
    # with open(files[test_index]) as f:
    #     test = json.load(f)
    # BASE_MODEL = "./llamas/LLaMA-7B-HF/"
    # BASE_MODEL = "./llamas/LLaMA-13B-HF/"
    # BASE_MODEL = "/media/backup/michaelf/ckpts/llamas/LLaMA-30B-HF/"

    #===

    TEST = """
       {}       
       """.format(examples[test_index]["question"])

    #===

    PROMPT = BASE_PROMPT + TEST
    inputs = tokeniser(PROMPT, return_tensors = "pt")
    input_ids = inputs["input_ids"].to("cuda:0")

    print(input_ids.size(), model.model.config.max_position_embeddings)

    # LLaMA17B: T = 0.1, K = 20, MAX_NEW_TOKENS = 128. Greedy decoding works well for LLaMA13B in FSL, hence with small T and K.

    # LLaMA30B: T = 0.5 and K = 20. Even with `NUM_SAMPLES` = 10, we get the right answer.

    total_gens = 0

    #===

    generated_sols_text = []
    generated_sols = []

    #===

    seq_metadata = []

    #===

    for n in range(NUM_SAMPLES):
        
        print("{}/{}".format(n + 1, NUM_SAMPLES))
        print("=" * 75)
        #print("\n")
        #===
        with torch.no_grad():
            generation_output = model.generate(input_ids = input_ids, do_sample = True, temperature = TEMP,
                                               max_new_tokens = MAX_NEW_TOKENS, top_k = TOP_K,
                                               return_dict_in_generate = True, output_scores = True)
        
        #===
            
        s = tokeniser.decode(generation_output.sequences[0][input_ids.size()[1]:])
        try:
            PATTERN_1 = "####"# (.*|\s)"
            PATTERN_2 = '\n\n'
            START = re.search(PATTERN_1, s).start()
            END = re.search(PATTERN_2, s[START:]).start()
            #===
            template = s[START+4: START+END] # Do not include "." at the end of sentence.
            # S, E = re.search("(\$).*", template).start(), re.search("(\$).*", template).end()
            # num = template[S: E].replace("\\boxed{", "").replace("}$", "$")
            num = template.replace(" ", "").replace(",", "")
            print("-" * 75)
            print("=" * 75)
            print("\n")
            total_gens = total_gens + 1
            generated_sols.append(num)
            
            #=== Estimate P(sequence).
            
            transition_scores = model.compute_transition_scores(generation_output.sequences, 
                                                                generation_output.scores, 
                                                                normalize_logits = True)
            #===
            input_length = input_ids.size()[1]
            generated_tokens = generation_output.sequences[:, input_length:]
            #=== Stop tracking Probas after token 4311 = "}$."
            running_tokens = []
            probas = []
            
            print("{:5s} | {:11s} | {:4s} | {:2s}".format("Token ID", "Token", "Logit (log P)", "P(Token | PTs)"))
            print("=" * 75)
            
            print("\n")
            
            num_tokens = 0
            
            for tok, score in zip(generated_tokens[0], transition_scores[0]):
                
                # | token | token string | logits (= log of P) | probability
                print(f"| {tok:5d} | {tokeniser.decode(tok):11s} | {score.cpu().numpy():.4f} | {np.exp(score.cpu().numpy()):.2%}")
        
                log_p = score.cpu().numpy()
        
                probas.append(log_p)
                #===
                num_tokens = num_tokens + 1
                #===
                running_tokens.append(tok.item())
                
                #=== box = 1884, ed = 287. make sure the "}$." + comes after "Final answer: The final answer..."
                #=== final answer is ---> [2186, 1234, 338].
                
                if tok == 4311 and set([1884, 287, 2186, 1234, 338]).issubset(set(running_tokens)):
                    #print("HERE!")
                    break
                
                #=== Solution might be $\boxed{\sqrt{2}}$; 26253 = }}$.
                if tok == 26253 and set([1884, 287, 2186, 1234, 338]).issubset(set(running_tokens)):
                    
                    break
            
            #===
            
            # We are now using \sum\_{n} log(P_{n})
            
            p = np.array(probas).sum() #
            
            #===
            
            d = {}
            
            d["Solution"], d["Sequence"], d["P(Seq)"] = num, re.split(PATTERN_1, s)[0] + s[START: END], str(p)
            
            d["Norm P(Seq)"] = str(p/num_tokens)
            
            seq_metadata.append(d)
            
        except AttributeError:
            
            pass
        
        #break
        print("\n")

    #===

    print("\n")
    counter = Counter(generated_sols)
    with open(LOG_PATH / Path(f'{test_index}_sol.json'), "w") as f:
        json.dump([dict([(key, str(val)) for key, val in data.items()]) for data in seq_metadata], f)
       

        for element in counter.most_common():
            
            value, count = element[0], element[1]
            p = (count/total_gens) * 100
            
            print("Value: {} | Count: {}/{} | Confidence: {:.1f}%".format(value, count, NUM_SAMPLES, p))
            json.dump(["Value: {} | Count: {}/{} | Confidence: {:.1f}%".format(value, count, NUM_SAMPLES, p)], f)