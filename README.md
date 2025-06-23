# LAMAR
A Foundation **La**nguage **M**odel for RN**A** **R**egulation

This repository contains codes and links of pre-trained weights for RNA foundation language model **LAMAR**. LAMAR outperformed benchmark models in various RNA regulation tasks, helping to decipher the regulation rules of RNA.  

LAMAR was developed by Rnasys Lab and Bio-Med Big Data Center, Shanghai Institute of Nutrition and Health (SINH), Chinese Academy of Sciences (CAS).  
![image](./ReadMe/overview.png)

## Citation
https://www.biorxiv.org/content/10.1101/2024.10.12.617732v2

## Create environment
The environment can be created with `LAMAR_requirements.txt`.  
```shell
git clone https://github.com/zhw-e8/LAMAR.git
cd ./LAMAR

conda create -n lamar python==3.11
conda activate lamar
pip install -r LAMAR_requirements.txt
```

The pretraining was conducted on A800 80GB GPUs, and the fine-tuning was conducted on the Sugon Z-100 16GB and Tesla V100 32GB clusters of GPUs.  
The environments are a little different on different devices.   
Pretraining environment:   
    A800: environment_A800_pretrain.yml  
Fine-tuning environment:   
    Sugon Z-100: environment_Z100_finetune.yml  
    V100(ppc64le): environment_V100_finetune.yml

### Required packages
accelerate >= 0.26.1  
torch >= 1.13  
transformers >= 4.32.1  
datasets >= 2.12.0  
pandas >= 2.0.3  
safetensors >= 0.4.1  

## Usage
### Install package
From github
```shell
pip install .
```
### Download pretrained weights
The pretrained weights can be downloaded from https://huggingface.co/zhw-e8/LAMAR/tree/main.

### Compute embeddings
```python
from LAMAR.modeling_nucESM2 import EsmModel
from transformers import AutoConfig, AutoTokenizer
from safetensors.torch import load_file, load_model
import torch


seq = "ATACGATGCTAGCTAGTGACTAGCTGATCGTAGCTG"
model_max_length = 1026
device = torch.device("cuda:0")
# instance tokenizer and config
tokenizer = AutoTokenizer.from_pretrained("tokenizer/single_nucleotide/", model_max_length=model_max_length)
config = AutoConfig.from_pretrained(
    "config/config_150M.json", vocab_size=len(tokenizer), pad_token_id=tokenizer.pad_token_id,
    mask_token_id=tokenizer.mask_token_id, token_dropout=False, positional_embedding_type='rotary', 
    hidden_size=768, intermediate_size=3072, num_attention_heads=12, num_hidden_layers=12
)
# intance the model and load pretrained weights
model = EsmModel(config)
weights = load_file('pretrain/saving_model/mammalian80D_4096len1mer1sw_80M/checkpoint-250000/model.safetensors')
weights_dict = {}
for k, v in weights.items():
    new_k = k.replace('esm.', '') if 'esm' in k else k
    weights_dict[new_k] = v
model.load_state_dict(weights_dict, strict=False)
model = model.to(device)
# Compute embeddings
model.eval()
with torch.no_grad():
    inputs = tokenizer(seq, return_tensors="pt")
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    outputs = model(
        input_ids=input_ids, 
        attention_mask=attention_mask
    )
    embedding = outputs.last_hidden_state[0, 1 : -1, :]
```
The paths of scripts:   
    Compute embeddings of nucleotides: src/embedding/NucleotideEmbeddingMultipleTimes.ipynb  
    Compute embeddings of functional elements: src/embedding/FunctionalElementEmbedding.ipynb  
    Compute embeddings of transcripts: src/embedding/RNAEmbedding.ipynb  
    Compute embeddings of splice sites: src/embedding/SpliceSiteEmbedding.ipynb  

### Predict splice sites from pre-mRNA sequences
The paths of scripts:  
    Tokenization: src/SpliceSitePred/tokenize_data.ipynb  
    Fine-tune: src/SpliceSitePred/finetune.ipynb

### Predict the translation efficiencies of mRNAs based on 5' UTRs (HEK293 cell line)
The paths of scripts:   
  Tokenization: src/UTR5TEPred/tokenize_data.ipynb  
  Fine-tune: src/UTR5TEPred/finetune.ipynb  
  Evaluate: src/UTR5TEPred/evaluate.ipynb  

### Annotate the IRES
The paths of scripts:  
  Tokenization: src/IRESPred/tokenize_data.ipynb  
  Fine-tune: src/IRESPred/finetune.ipynb  
  Evaluate: src/IRESPred/evaluate.ipynb  
  
### Predict the half-lives of mRNAs based on 3' UTRs (BEAS-2B cell line)  
The paths of scripts:   
  Tokenization: src/UTR3DegPred/tokenize_data.ipynb  
  Fine-tune: src/UTR3DegPred/finetune.ipynb  
  Evaluate: src/UTR3DegPred/evaluate.ipynb  

## Baseline methods
The performance of LAMAR was compared to baseline methods. The scripts: https://github.com/zhw-e8/LAMAR_baselines
