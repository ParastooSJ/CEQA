from torch.utils.data import DataLoader
import math
from sentence_transformers import LoggingHandler, util
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CESoftmaxAccuracyEvaluator,CECorrelationEvaluator,CEBinaryAccuracyEvaluator,CERerankingEvaluator
from sentence_transformers.evaluation import MSEEvaluator, SentenceEvaluator
from sentence_transformers import InputExample
import logging
from datetime import datetime
import os
import gzip
import csv
import json
import numpy as np
import torch
import sys
import random



class RecallEvaluator(SentenceEvaluator):
    """
    This evaluator computes recall at k for a CrossEncoder model.
    """
    def __init__(self, samples, k=20, name=''):
        """
        samples: A list of dictionaries, each containing three keys: 'query', 'positive', and 'negative'.
        Each key maps to a list of passages (texts).
        k: The number of top-ranked items to consider for computing recall.
        name: A name for the evaluator.
        """
        self.samples = samples
        self.k = k
        self.name = name
        #self.output_path = output_path

    def __call__(self, model,output_path=None, epoch=-1, steps=-1):
        #model.eval()
        recall_scores = []

        with torch.no_grad():
            for sample in self.samples:
                
                query = sample['query']
                positive_passages = sample['positive']
                negative_passages = sample['negative']
                passages = positive_passages + negative_passages
                labels = [1] * len(positive_passages) + [0] * len(negative_passages)

                # Create pairs and predict
                pairs = [(query, passage) for passage in passages]
                scores = model.predict(pairs)

                # Get the indices of the top k scores
                top_k_indices = np.argsort(scores)[::-1][:self.k]

                # Calculate the number of relevant documents in the top k
                num_relevant_top_k = sum(labels[idx] for idx in top_k_indices)

                # Calculate recall for this sample and append to the list
                if len(positive_passages) ==0:
                    recall = 0
                else:
                    recall = num_relevant_top_k / len(positive_passages)
                recall_scores.append(recall)

        # Calculate the average recall over all samples
        average_recall = np.mean(recall_scores)
        print(f"Average Recall@{self.k}: {average_recall:.2f}")
        print("----------------")

        
        return average_recall



def make_samples(file):
    threshold = 0.0
    samples = []
    selected_samples = []
    selected_number = 16 * 1000
    has_ev = 0
    for sample in file:
        question = sample["question"]
        sentence1 = question
        if "selected_evidences" not in sample:
            sample["selected_evidences"] = {}
        
        if len(sample["selected_evidences"])>0:
            evidence = next(iter(sample["selected_evidences"]))
            value = sample["selected_evidences"][evidence]
            if value > threshold:
                sentence1 = question + " evidence: "+ evidence
                has_ev +=1
        
        for passage in sample["ctxs"][:10]:
            passage_text = passage["text"]
            label = 1 if passage["has_answer"] else 0
            
            samples.append(InputExample(texts=[sentence1, passage_text], label=label))

    selected_samples = random.sample(samples, selected_number)
   
    return samples ,selected_samples

def make_samples_dev(file):
    samples = []
    
    for sample in file:
        try:
            
            answers = sample["answers"]
            question = sample["question"]
            if "selected_evidences" in sample.keys():
                if len(sample["selected_evidences"])>0:
                    first_key = next(iter(sample["selected_evidences"]))
            else:
                first_key = ""
            sentence1 = question +" evidence:"+first_key
            negative_list = []
            positive_list = []

            for passage in sample["ctxs"][:100]:
                passage_text = passage["text"]
                label = 1 if passage["has_answer"] else -1
                
                if label==1:
                    positive_list.append(passage_text)
                else:
                    negative_list.append(passage_text)

            inp = {"query":sentence1, "positive":positive_list, "negative":negative_list}
            samples.append(inp)
            break
                
        except:
            print('error')
    return samples
    

def custom_collate(batch):
    texts = [example.texts for example in batch]
    labels = [example.label for example in batch]
    return {'texts': texts, 'labels': labels}


def make_samples_test(file):
    samples = []
    
    
    for sample in file:
        try:
            if "selected_evidences" not in sample.keys():
                sample["selected_evidences"] = {}
            question = sample["question"]
            if len(sample["selected_evidences"])>0:
                first_key = next(iter(sample["selected_evidences"]))
                value = sample["selected_evidences"][first_key]
                
            else:
                first_key = ""
            if first_key=="":
                sentence1 = question
            else:
                sentence1 = question +" evidence: "+first_key
            
            for passage in sample["ctxs"][:1000]:
                passage_text = passage["text"]
                label = 0
                inp = InputExample(texts=[sentence1, passage_text], label=label)
                samples.append(inp)
        except:
            print('error')
    return samples


def train(model_save_path, train_path, dev_path):
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


    train_samples, train_selected = make_samples(train_f)
    dev_samples = make_samples_dev(dev_f)

    train_batch_size = 16
    num_epochs = 1
    
    #Define our CrossEncoder model. We use distilroberta-base as basis and setup it up to predict 3 labels

    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2',num_labels=1,max_length=512)
    #We wrap train_samples, which is a list of InputExample, in a pytorch DataLoader
    train_dataloader = DataLoader(train_selected, shuffle=True, batch_size=train_batch_size)

    #During training, we use CESoftmaxAccuracyEvaluator to measure the accuracy on the dev set.
    
    recall_evaluator = RecallEvaluator(dev_samples, k=10, name='dev-evaluator')


    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
    logger.info("Warmup-steps: {}".format(warmup_steps))


    # Train the model
    model.fit(train_dataloader=train_dataloader,
            evaluator=recall_evaluator,
            epochs=num_epochs,
            evaluation_steps=1000,
            warmup_steps=warmup_steps,
            output_path=model_save_path)

    

def test(model_path, test_path, out_file_path):
    if not os.path.exists(model_path):
        print(f"Error: Model path '{model_path}' does not exist.")
        return
    test_batch_size = 100
    
    model = CrossEncoder(model_path)
   
    test_f = json.load(open(test_path,'r'))
    
    test_samples = make_samples_test(test_f)
    
    test_dataloader = DataLoader(test_samples, shuffle=True, batch_size=test_batch_size, collate_fn=custom_collate)

    all_test_scores = {}
    i = 0
    for batch in test_dataloader:
        i+=1
        print(i)
        
        texts_batch = batch["texts"]
        batch_scores = model.predict(texts_batch)
        
        for sample,score in zip(texts_batch,batch_scores):
            question = sample[0].split(' evidence:')[0]
            print(score)
            all_test_scores[question+"_"+sample[1]] = score
        

    data = []
    for i,line in enumerate(test_f):
        
        question = line["question"]
        passages = []
        for passage in line["ctxs"][:1000]:
            try:
                passage_text = passage["text"]
                score = all_test_scores[question+"_"+passage_text]
                passage["score"] = float(score)
                passages.append(passage)

            except Exception as e:
                print(e)
        line["ctxs"] = passages
        data.append(line)
    with open(out_file_path,'w') as f:
        json.dump(data,f, indent=2)

def construct_paths(dataset_name, retriever_name):
    base_path = "../"
    data_base = f"{base_path}/data/{dataset_name}"
    model_save_base = f"{base_path}/model/{dataset_name}_Reranker"

    paths = {
        "train": f"{data_base}/processed_train.json",
        "dev": f"{data_base}/processed_train.json",
        "test": f"{data_base}/{retriever_name}/test-passage-evidence-ranked.json",  # Adjust path for test data
        "model_save": model_save_base,
        "out_file": f"{data_base}/{retriever_name}/trained-ms-marco-MiniLM-1000-nothreshold-scored.json"
    }
    return paths


def main():
    if len(sys.argv) != 4:
        print("Usage: python script.py <train/test> <dataset_name> <retriever_name>")
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
