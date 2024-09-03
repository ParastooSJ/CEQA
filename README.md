# Contextual Evidence-based Question Answering
The Contextual Evidence-Driven Question Answering (CEQA) framework revolutionizes Open-Domain Question Answering (ODQA) by integrating Textual Knowledge Graphs (TKGs) to enhance the accuracy and relevance of generated answers. Traditional ODQA systems typically rely on a two-step process involving retrieval and reading modules. CEQA extends this approach by employing TKGs, which are semi-structured graphs with Wikipedia documents as nodes and sentences as edges, to extract and rank contextual evidence. This evidence is then used to re-rank passages retrieved by conventional algorithms, ensuring that the most contextually relevant and informative passages are prioritized. The system's core lies in its ability to capture and utilize the nuanced relationships between entities in the text, thereby improving the precision of both passage retrieval and answer generation.

The CEQA framework introduces several key components: an evidence extraction and ranking module, a passage retrieval module, an evidence-based passage re-ranking module, and an evidence-based generative reader. Initially, the evidence extraction module uses TKGs to identify and rank the most relevant contextual relationships related to the question. This is followed by the passage retrieval module that retrieves potential passages, which are subsequently re-ranked using the top-ranked evidence from the TKGs. Finally, the evidence-based generative reader synthesizes this evidence with the retrieved passages to generate accurate and contextually rich answers. This integrated approach not only enhances the quality of evidence used for answering queries but also mitigates the risk of generating erroneous or "hallucinated" content, setting a new standard for ODQA systems.

## DATA
The data for TriviaQA, SQuAD Open, and NQ with extracted evidences from textual knowledge graph can be downloaded [here](https://drive.google.com/drive/folders/18PgPdFA_34L6RdBeeZNT1E_r0JvE---0?usp=share_link). Unzip and place it in Data folder. We have provided the top-1000 passages retrieved by Mss-DPR, DPR, MSS, BM25 and Contriever for each dataset.

## Pretrained Models
The pretrained models for both Erank and Ereader, can be found [here](https://drive.google.com/drive/folders/1j3FPAKciB89X-H-mgChgYcO5erATP5qs?usp=share_link). Unzip and place it in model folder.

## Test
1. Download the data for the selected dataset from [here](https://drive.google.com/drive/folders/18PgPdFA_34L6RdBeeZNT1E_r0JvE---0?usp=share_link) and place them Under Data/{dataset_name}. If you wish to use the pretrained models, make sure to download them and place them in Model folder.
2. To start, the model needs to rank evidences utilizing the pretrained evidence ranker model, to do so run the following command for the selected dataset and retriever:
```
cd src/
python evidence_ranker.py test <dataset_name> <retriever_name>
``` 
3. To get the top-100 passages from the evidence-based reranker for the selected retriever, run the following command:
```
python PassageRankerModel.py test {dataset_name} {retriever_name}
```
4. To generate the Input file for Generative evidence-based Reader, run the following command:
```
python prepare_fid_input.py {dataset_name} {retriever_name}
```
5. After the input files are prepared, run the following commands to generate the answers for the selected dataset.
```
mkdir FiD
cd FiD
git clone https://github.com/facebookresearch/FiD.git
cd FiD
python test_reader.py \
        --model_path path/to/the/pretrained/reader/model/for/the/selected/reader/model \
        --eval_data path/to/the/prepared/test/file \
        --per_gpu_batch_size 1 \
        --n_context 100 \
        --name my_test \
        --checkpoint_dir checkpoint \
```
