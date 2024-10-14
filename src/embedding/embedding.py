from LaMorena.modeling_nucESM2 import EsmModel
from transformers import AutoConfig, AutoTokenizer
import random
import tqdm
import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file, load_model


def model_selection(model_state_path, model_max_length):
    """
    Instance tokenizer and model loading pretrained weights or not
    Params:
        model_state_path: Path of pretrained weight; = None if training or inference from scratch
        model_max_length: Max length of tokenizer
    Return:
        tokenizer
        model
    """
    tokenizer = AutoTokenizer.from_pretrained("tokenizer/single_nucleotide/", model_max_length=model_max_length)
    config = AutoConfig.from_pretrained(
        "config/config_150M.json", vocab_size=len(tokenizer), pad_token_id=tokenizer.pad_token_id, mask_token_id=tokenizer.mask_token_id, token_dropout=False, positional_embedding_type='rotary', 
        hidden_size=768, intermediate_size=3072, num_attention_heads=12, num_hidden_layers=12
    )
    model = EsmModel(config)
    if model_state_path:
        weights = load_file(model_state_path)
        # ['MODEL_STATE']
        weights_dict = {}
        for k, v in weights.items():
            new_k = k.replace('esm.', '') if 'esm' in k else k
            weights_dict[new_k] = v
        model.load_state_dict(weights_dict, strict=False)
        
    return tokenizer, model


def seq2embedding(seqs, embedding_type, model_state_path, model_max_length, return_embedding, gpu_i=0):
    """
    Compute the means of embeddings of sequences
    Params:
        seqs: Input sequences
        embedding_type: nucleotide or sequence
        model_state_path: Path of pretrained weight; = None if training or inference from scratch
        model_max_length: Max length of tokenizer
        return_embedding: Return embeddings of sequences or not
        gpu_i: Index of GPU
        
    Return:
        embedding_means: Means of embeddings, 2-dimension list
        embeddings: Embeddings, 3-dimension list, Nseq * Ntoken * hidden size
    """
    device = torch.device("cuda:{}".format(gpu_i))
    tokenizer, model = model_selection(model_state_path, model_max_length)
    model = model.to(device)
    embeddings = []
    embedding_means = []
    model.eval()
    with torch.no_grad():
        for seq in tqdm.tqdm(seqs):
            inputs = tokenizer(seq, return_tensors="pt")
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask
            )
            outputs = outputs.last_hidden_state[0, 1 : -1, :]
            if embedding_type == 'nucleotide':
                embeddings.append(outputs.tolist())
            elif embedding_type == 'sequence':
                if return_embedding:
                    embeddings.append(outputs.tolist())
                outputs_mean = torch.mean(outputs, dim=0)
                embedding_means.append(outputs_mean.tolist())
                
    return embeddings, embedding_means