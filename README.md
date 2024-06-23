# Contextual Evidence-based Question Answering
## DATA
The data for TriviaQA, SQuAD Open, and NQ can be downloaded [here](https://drive.google.com/drive/folders/18PgPdFA_34L6RdBeeZNT1E_r0JvE---0?usp=share_link). Unzip and place it in model folder.

## Pretrained Models
The pretrained models for both Erank and Ereader, can be found [here](https://drive.google.com/drive/folders/1j3FPAKciB89X-H-mgChgYcO5erATP5qs?usp=share_link). Unzip and place it in model folder.

## Code
1. Download the data for the selected dataset from [here](https://drive.google.com/drive/folders/18PgPdFA_34L6RdBeeZNT1E_r0JvE---0?usp=share_link) and place them Under Data/{dataset_name}. If you wish to use the pretrained models, make sure to download them and place them in Model folder.
2. To get the top-100 passages from the evidence-based reranker, run the following command:
''' cd src/
python PassageRankerModel.py test {dataset_name}
'''
3. To generate the Input file for Generative evidence-based Reader, run the following command:
'''python prepare_fid_input.py {dataset_name}'''
4. After the input files are prepared, run the following commands to generate the answers for the selected dataset.
''' git clone  
