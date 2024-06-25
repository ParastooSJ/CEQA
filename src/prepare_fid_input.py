import json

import sys


dataset_name = sys.argv[1]
retriver_name = sys.argv[2]

train_path = f'../data/{dataset_name}/processed_train.json'
dev_path = f'../data/{dataset_name}/{retriver_name}/trained-ms-marco-MiniLM-1000-nothreshold-scored.json'
test_path = f'../data/{dataset_name}/{retriver_name}/trained-ms-marco-MiniLM-1000-nothreshold-scored.json'

train_f = json.load(open(train_path,'r'))
dev_f = json.load(open(dev_path,'r'))
test_f  = json.load(open(test_path,'r'))

inputs = [train_f,dev_f,test_f]
#inputs = [test_f]


train_out_path = f'../data/{dataset_name}/mytrain.json'
dev_out_path = f'../data/{dataset_name}/mydev.json'
test_out_path = f'../data/{dataset_name}/mytest-{retriver_name}.json'


out_paths = [train_out_path,dev_out_path,test_out_path]

for f, out in zip(inputs,out_paths):
    data = []
    for line in f:
        if "selected_evidences" not in line.keys():
            line["selected_evidences"] = {}
        if len(line["selected_evidences"])>0:
            first_key = next(iter(line["selected_evidences"]))
            
            
            line["question"] = line["question"] +" evidence: "+first_key
            
        try:
            line["ctxs"] = sorted(line["ctxs"], key=lambda x: x['combined_score'], reverse=True)[:100]
        except:
            line["ctxs"] = sorted(line["ctxs"], key=lambda x: x['score'], reverse=True)[:100]
        data.append(line)
    
    with open(out,'w') as f_out:
        json.dump(data, f_out, indent=2)
