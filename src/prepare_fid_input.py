import json

threshold = 0.7
#train_path = "/home/jparastoo/downloads/ODQA/data/TQA/processed_train.json"
#dev_path = "/home/jparastoo/downloads/ODQA/Final_data/TQA/mss-dpr/trained-ms-marco-MiniLM-1000-nothreshold-scored.json"
test_path = "/home/jparastoo/downloads/ODQA/Final_data/SQUAD/contriver/trained-ms-marco-MiniLM-1000-nothreshold-scored.json"

#train_f = json.load(open(train_path,'r'))
#dev_f = json.load(open(dev_path,'r'))
test_f  = json.load(open(test_path,'r'))

inputs = [test_f]#[train_f,dev_f,test_f]
#inputs = [test_f]
#train_out_path = "/home/jparastoo/downloads/ODQA/data/NQ/train_fid_with_evidence.json"
#dev_out_path = "/home/jparastoo/downloads/ODQA/data/NQ/dev_fid_with_evidence.json"
#test_out_path = "/home/jparastoo/downloads/ODQA/data/NQ/test_fid_with_evidence.json"

#train_out_path = "/home/jparastoo/downloads/FID/FiD/open_domain_data/TQA/mytrain.json"
#dev_out_path = "/home/jparastoo/downloads/FID/FiD/open_domain_data/TQA/mydev.json"
test_out_path = "/home/jparastoo/downloads/FID/FiD/open_domain_data/Squad/mytest-contriver.json"


out_paths = [test_out_path] #[train_out_path,dev_out_path,test_out_path]

for f, out in zip(inputs,out_paths):
    data = []
    for line in f:
        if "selected_evidences" not in line.keys():
            line["selected_evidences"] = {}
        if len(line["selected_evidences"])>0:
            first_key = next(iter(line["selected_evidences"]))
            
            #if line["selected_evidences"][first_key]>threshold:
            line["question"] = line["question"] +" evidence: "+first_key
            
        try:
            line["ctxs"] = sorted(line["ctxs"], key=lambda x: x['combined_score'], reverse=True)[:100]
        except:
            line["ctxs"] = sorted(line["ctxs"], key=lambda x: x['score'], reverse=True)[:100]
        data.append(line)
    threshold = 0.5
    with open(out,'w') as f_out:
        json.dump(data, f_out, indent=2)
