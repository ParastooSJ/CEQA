import json
def calculate_mrr_at_10(ranked_lists):
    print(len(ranked_lists))
    print('------')
    total_mrr = 0.0
    num_queries = len(ranked_lists)

    for ranked_list in ranked_lists:
        reciprocal_rank = 0.0

        for i, document in enumerate(ranked_list[:10], start=1):
            if document==1:
                reciprocal_rank = 1.0 / i
                break

        total_mrr += reciprocal_rank

    # Calculate the mean MRR at 10
    mean_mrr_at_10 = total_mrr / num_queries

    return mean_mrr_at_10


def calculate_recalls(ranked_lists, max_rank=100):
    # Initialize an array to hold recall values for each rank from 1 to max_rank
    recalls = [0.0] * max_rank
    num_queries = len(ranked_lists)

    # Calculate recall for each rank
    for rank in range(1, max_rank + 1):
        all_found = 0
        for ranked_list in ranked_lists:
            # Check if the relevant document (marked as '1') is within the top 'rank' positions
            if 1 in ranked_list[:rank]:
                all_found += 1
        # Calculate recall for the current rank
        recall = all_found / num_queries
        recalls[rank - 1] = recall *100

    return recalls



def calculate_recall_at_10(ranked_lists,rank):
    total_recall = 0.0
    num_queries = len(ranked_lists)

    all_found = 0
    for ranked_list in ranked_lists:
        print(len(ranked_list))
        if 1 in ranked_list[:rank]:
            all_found+=1
        recall = all_found / num_queries
         

    # Calculate the mean Recall at 10
    

    return recall
# Example usage:
# Assuming you have a list of ranked documents for each query, each with a boolean 'is_relevant' field
# Replace this with your actual data structure

#jfile = json.load(open('/home/jparastoo/downloads/ODQA/data/TQA/trained-ms-marco-MiniLM-1000-nothreshold-scored.json','r'))

#jfile = json.load(open('/home/jparastoo/downloads/ODQA/Final_data/TQA/mss/trained-ms-marco-MiniLM-1000-nothreshold-scored.json','r'))
jfile = json.load(open('/home/jparastoo/downloads/UPR/unsupervised-passage-reranking/downloads/data/retriever-outputs/mss-dpr/reranked/trivia-test.json','r'))
#jfile = json.load(open('/home/jparastoo/downloads/ODQA/data/NQ/test_first_best.json','r'))
#jfile = json.load(open('/home/jparastoo/downloads/ODQA/data/NQ/test_evidence_golden_regression.json','r'))
#jfile = json.load(open('/home/jparastoo/downloads/ODQA/data/TQA/test_evidence_golden.json','r'))
#jfile  = json.load(open('/home/jparastoo/downloads/UPR/unsupervised-passage-reranking/downloads/data/retriever-outputs/mss-dpr/reranked/nq-test.json','r'))

ranked_lists = []
for line in jfile:
    ranked_list = []
    answers = line["answers"]
    try:
        line["ctxs"] = sorted(line["ctxs"], key=lambda x: x['score'], reverse=True)
    except:
        x=1
    all_labels = 0
    for context in line["ctxs"]:
        label = 0
        if "has_answer" not in context.keys():
            for answer in answers:
                if answer in context["text"]:
                    label =1
                    all_labels +=1
        else:
            if context["has_answer"]:    
                label = 1
                all_labels +=1
        #if context["has_answer"] and label==0:
         #   print(context["text"])
          #  print(answers)
           # print('-----')
       
        ranked_list.append(label)
    ranked_lists.append(ranked_list)
    

mean_mrr_at_10 = calculate_mrr_at_10(ranked_lists)
recall_1 = calculate_recall_at_10(ranked_lists,1)
recall_10 = calculate_recall_at_10(ranked_lists,10)
recall_100 = calculate_recall_at_10(ranked_lists,100)
recalls = calculate_recalls(ranked_lists)


print(f"Mean Reciprocal Rank at 10: {mean_mrr_at_10:.4f}")
print(f"Recall at 1: {recall_1:.4f}")
print(f"Recall at 10: {recall_10:.4f}")
print(f"Recall at 100: {recall_100:.4f}")


print(recalls)