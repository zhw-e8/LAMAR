# LAMAR
A Foundation Language Model for Multilayer Regulation of RNA

This repository contains code and pre-trained weights for RNA foundation language model **LAMAR**. LAMAR outperformed benchmark models in various RNA regulation tasks, helping to decipher the regulation rules of RNA.  

La Morena was developed by Rnasys Lab and Bio-Med Big Data Center, Shanghai Institute of Nutrition and Health (SINH), Chinese Academy of Sciences (CAS).  
![image](./ReadMe/fig1.png)

## Citation

## Environment
The pretraining was conducted on A800 80GB graphic process units, and the fine-tuning was conducted on the Sugon Z-100 16GB and Tesla V100 32GB clusters of graphic process units.  
The environments are a little different on different devices. **The unified environment will be developed.**   
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

### Compute embeddings
```python

model.eval()
```

### Predict splice sites from pre-mRNA sequences

### Predict the translation efficiency of mRNAs based on 5' UTRs (HEK293 cell line)
The paths of scripts:   
  Fine-tune: src/UTR5TEPred/Timothy/finetune/finetune.ipynb  
  Evaluate: src/UTR5TEPred/Timothy/finetune/evaluate.ipynb

### Predict the half-time of mRNAs based on 3' UTRs (cell line)
