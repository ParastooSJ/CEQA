
import json
import re
from collections import OrderedDict
from operator import itemgetter
from sentence_transformers import CrossEncoder,SentenceTransformer, util
import numpy as np
import pandas as pd
import sys
import torch
import random
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to(device)
class DataPrep():

    def select_triples(self,line):
        
        
        max_evidence = 100 if len(line["evidences"])>100 else len(line["evidences"])
        
        sorted_dict = dict(sorted(line["scored_evidences"].items(), key=lambda item: item[1], reverse=True))
        top_max_evidence = dict(list(sorted_dict.items())[:max_evidence])
        
        return top_max_evidence

    def score_sample(self,line):

        if "target" in line:
            answers =[ line["target"] ]
        else:
            answers = line["answers"]

        encoded_dict = self.process_batch(line)
        question = line["question"]
        
        max_score = float('-inf')
        
        evidences = line['evidences']

        score_evidence = {}
        for evidence in evidences:
            en_evidence = encoded_dict[evidence]
            max_score = float('-inf')
            for answer in answers:
                en_answer = encoded_dict[answer]
            
                score = util.pytorch_cos_sim(en_evidence,en_answer)
                score = float(score[0][0])
                if score> max_score:
                    max_score = score
            
            score_evidence[evidence] = max_score
        line["scored_evidences"] = score_evidence
        line["selected_evidences"] = self.select_triples(line)
        
        return line

    
    

    def process_batch(self,sample):
        
        in_list = set()

        in_list.add(sample["question"])

        if "target" in sample:
            answer = sample["target"]
            in_list.add(answer)
        else:
            answers = sample["answers"]
            in_list.update(answers)
        in_list.update(sample["evidences"])
        
        encoded_batch = model.encode([sample for sample in in_list])
        
        results = {}
        for key, value in zip(in_list,encoded_batch):
            results[key] = value
            
        return results


def process_line(line,count):
    
    dp = DataPrep()
    if "evidences" not in line.keys():
        line["evidences"] = []
    finilized_sample = dp.score_sample(line)

    print(count)
    return finilized_sample
    

def main(dataset_name):
    input_path = f'../data/{dataset_name}/processed_train.json'
    output_path = f'../data/{dataset_name}/train_evidence_scored.json'

    with open(input_path, 'r') as file_r:
        data = json.load(file_r)

    count = 0
    final_result = []

    num_threads = 100

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for line in data:
            try:
                count += 1
                if count > file_count:
                    future = executor.submit(process_line, line, count)
                    futures.append(future)

                if count >= file_count + 90000:
                    break

            except Exception as e:
                print(f'Error processing line {count}: {e}')

        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                final_result.append(result)
            except Exception as e:
                print(f'Error processing future: {e}')

    with open(output_path, 'w') as f:
        json.dump(final_result, f, indent=2)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python score_evidence.py <dataset_name>")
        sys.exit(1)
    
    dataset_name = sys.argv[1]
    main(dataset_name)





            
        
