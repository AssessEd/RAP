import numpy as np
import heapq 
from difflib import SequenceMatcher
import json
import gzip
from functools import cache

TRAIN_FILE = "./data/gsm8k/train.jsonl"
TEST_FILE = "./data/gsm8k/test.jsonl"
RES_FILE = "./data/gsm8k/knn_{}.jsonl"

zipped_dict = dict()

@cache 
def encode_str(problem_str):
    return len(gzip.compress(problem_str.encode()))

def compute_gzip_distance(problem, reference):
    x1 = encode_str(problem)
    x2 = encode_str(reference)     
    return (encode_str(problem + " " + reference) - min(x1, x2)) / max(x1, x2)

distance_fcn = dict(
    sequence_matcher=lambda problem, reference: SequenceMatcher(None, problem, reference).ratio(),
    gzip=compute_gzip_distance
)


if __name__ == "__main__":
    NUM_SHOTS = 20
    TYPE = 'sequence_matcher'
    TYPE = 'gzip'

    with open(TRAIN_FILE) as f:
        train_data = [json.loads(l) for l in f]
    with open(TEST_FILE) as f:
        test_data = [json.loads(l) for l in f]
    print(len(train_data), len(test_data))
    RES_FILE = RES_FILE.format(TYPE)
    distance = distance_fcn[TYPE]
    #===
    k_nns = dict()
    for idx, test_example in enumerate(test_data):
        
        print("{}/{}".format(idx + 1, len(test_data)))
        print("-" * 105)
        
        reference = test_example["question"]
        target = test_example["answer"]
        
        #=== Find `NUM_SHOTS` most similar in training set.
        
        # ratios = []
        res = []
        for jdx, train_example in enumerate(train_data):
            
            problem = train_example["question"]

            ratio = distance(problem, reference)
            #=== Keep only top k elements
            if res and ratio > res[0][0] or len(res) < NUM_SHOTS:
                if len(res) >= NUM_SHOTS:
                    heapq.heappop(res)
                heapq.heappush(res, (ratio, jdx))
            # ratios.append(ratio)
        res.sort(reverse=True) # sorting heap for more effective storage

        #=== Sort `ratios`. NEED TO FILTER IF SIMILARITY = 1.
        # top_k = np.partition(ratios, -NUM_SHOTS)[-NUM_SHOTS:]

        # top_k = np.sort(top_k)[::-1]
        # print(top_k, [dist for dist, _ in res])
        #=== Train indices.

        # indexes = [idx for _, idx in res]
        k_nns[idx] = res 
    with open(RES_FILE, 'w') as f:
        json.dump(k_nns, f)
    with open(RES_FILE) as f:
        kk = json.load(f)
    print('dones')
    
        # indices = [ratios.index(element) for element in top_k] # bug, returns the same index for a repeat element
        # print(indices, indexes)
        #=== Create prompt.

        # PROMPT = ""

        # for c, index in enumerate(indexes):
        
        #     p, s = train_data[index]["question"], train_data[index]["answer"]
            
        #     p = p.replace("$", "\$")
        #     s = s.replace("$", "\$")
        
        #     #=== Same template as at training.
            
        #     STRING = "Problem:\n{}\n\nSolution:\n{}. ".format(p, s)
            
        #     PROMPT = PROMPT + STRING
            
        # #===
        
        # PROMPT = PROMPT + "Problem:\n{}\n\nSolution:\n".format(reference.replace("$", "\$"))