from torch.utils.data import DataLoader
import math
from sentence_transformers import LoggingHandler, util
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
from sentence_transformers import InputExample
import logging
import os
import gzip
import csv
import sys
import json
import numpy as np
import random


def make_samples(file):
    samples = []
    
    for sample in file:
        try:
            question = sample["question"]
            # Sort the selected_evidences by their value in descending order and take the first 10
            top_evidences = sorted(sample["selected_evidences"].items(), key=lambda item: item[1], reverse=True)[:10]
            for evidence, label in top_evidences:
                inp = InputExample(texts=[question, evidence], label=label)
                samples.append(inp)      
                
        except:
            print('error')
    print(len(samples))
    return samples
    
def custom_collate(batch):
    texts = [example.texts for example in batch]
    labels = [example.label for example in batch]
    return {'texts': texts, 'labels': labels}

def make_samples_test(file):
    samples = []
    
    for sample in file:
        try:
            question = sample["question"]
            for evidence in sample["evidences"]:
                inp = InputExample(texts=[question, evidence],label=0)
                samples.append(inp)
        except:
            print('error')
    return samples

def train(model_save_path,train_path,dev_path):
    steps = 10000
    train_batch_size = 16

    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
    logger = logging.getLogger(__name__)
    #### /print debug information to stdout


    logger.info("Read train dataset")

    

    train_f = json.load(open(train_path,'r'))
    
    dev_f = json.load(open(dev_path,'r'))

    train_samples = make_samples(train_f)
    dev_samples = make_samples(dev_f)

    random_train_samples = random.sample(train_samples,steps*train_batch_size)
   

    
    num_epochs = 1
    
    #Define our CrossEncoder model. We use distilroberta-base as basis and setup it up to predict 3 labels
    
    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2',num_labels=1,max_length=512)

    #We wrap train_samples, which is a list of InputExample, in a pytorch DataLoader
    train_dataloader = DataLoader(random_train_samples, shuffle=True, batch_size=train_batch_size)

    #During training, we use CESoftmaxAccuracyEvaluator to measure the accuracy on the dev set.
    
    
    evaluator = CECorrelationEvaluator.from_input_examples(dev_samples, name='dev')


    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
    logger.info("Warmup-steps: {}".format(warmup_steps))


    # Train the model
    model.fit(train_dataloader=train_dataloader,
            evaluator=evaluator,
            epochs=num_epochs,
            evaluation_steps=1000000,
            warmup_steps=warmup_steps,
            output_path=model_save_path)

def test(model_path,test_path,out_file_path):
    
    test_batch_size = 16
    model = CrossEncoder(model_path)

    test_f = json.load(open(test_path,'r'))
    

    test_samples = make_samples_test(test_f)
    
    test_dataloader = DataLoader(test_samples, shuffle=True, batch_size=test_batch_size, collate_fn=custom_collate)


    all_test_scores = {}

    for batch in test_dataloader:
        
        texts_batch = batch["texts"]
        batch_scores = model.predict(texts_batch)
        
        for sample,score in zip(texts_batch,batch_scores):
            
            all_test_scores[sample[0]+"_"+sample[1]] = score
        

   
    for i,line in enumerate(test_f):
        
        scored_evidences = {}
        question = line["question"]
        if 'evidences' in line.keys():
            for evidence in line["evidences"]:
                try:
                    score = all_test_scores[question+"_"+evidence]
                    scored_evidences[evidence] = float(score)
                except:
                    print('error')

            
            test_f[i]["scored_evidences"] = scored_evidences
             
        
            test_f[i]["selected_evidences"] = {k: v for k, v in sorted(test_f[i]["scored_evidences"].items(), key=lambda item: item[1], reverse=True)}
        
    
    with open(out_file_path,'w') as f:
        json.dump(test_f,f, indent=2)




def construct_paths(dataset_name, retriever_name):
    base_path = "../"
    data_base = f"{base_path}/data/{dataset_name}"
    model_save_base = f"{base_path}/model/{dataset_name}_evidence_ranker"

    paths = {
        "train": f"{data_base}/processed_train.json",
        "dev": f"{data_base}/processed_train.json",
        "test": f"{data_base}/{retriever_name}/test-passage-evidence.json",  # Adjust path for test data
        "model_save": model_save_base,
        "out_file": f"{data_base}/{retriever_name}/test-passage-evidence-ranked.json"
    }
    return paths


def main():
    if len(sys.argv) != 4:
        print("Usage: python evidence_ranker.py <train/test> <dataset_name> <retriever_name>")
        return

    mode, dataset_name, retriever_name = sys.argv[1], sys.argv[2], sys.argv[3]
    paths = construct_paths(dataset_name, retriever_name)

    if mode == "train":
        train(paths["model_save"], paths["train"], paths["dev"])
    elif mode == "test":
        test(paths["model_save"], paths["test"], paths["out_file"])
    else:
        print("Invalid mode. Choose 'train' or 'test'.")

if __name__ == "__main__":
    main()





