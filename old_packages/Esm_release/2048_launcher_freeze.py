import os
import json
import torch

import numpy as np
import pandas as pd
from scipy.special import softmax

from datasets import load_dataset, load_from_disk
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    # DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
from ESM.srcs.utils.metrics import METRICS_INDEX
from ESM.srcs.Applications.EsmForTokenClassifications.EsmForTokenClassification import EsmForTokenClassification
from ESM.srcs.Applications.EsmForTokenClassifications.EsmForTokenClassificationFrozen import EsmForTokenClassificationFrozen

def group_texts(tokenizer):
    def _inner(data):
        tokenized_inputs = tokenizer(
            data["seq"], truncation=True, max_length=tokenizer.model_max_length
        )
        return tokenized_inputs
    return _inner

def load_data(
    tokenizer, set_path=None, preload_data=None, seed=None, test_size=None,
):
    if os.path.exists(preload_data):
        data_for_preload = load_from_disk(preload_data)
    else:
        dataset= load_dataset("json", data_files=set_path)
        data_for_preload = dataset.map(
            group_texts(tokenizer), batched=True, num_proc=os.cpu_count()
            )
        data_for_preload=data_for_preload.rename_column("label","labels")
        data_for_preload=data_for_preload.remove_columns("seq")
        if not test_size is None:
            data_for_preload = data_for_preload["train"].train_test_split(
                test_size=test_size, shuffle=True, seed=seed
            )
        data_for_preload.save_to_disk(preload_data, num_proc=os.cpu_count())
    return data_for_preload

def load_metrics(methods):
    def compute_metrics(p):
        logits, labels = p
        logits=logits.reshape((logits.shape[0]*logits.shape[1],-1))
        table=pd.DataFrame(logits)
        table["pred"]=np.argmax(logits, axis=1)
        t_true=np.array(labels).flatten()
        print("finish flatten")
        result = {}
        for method_name, method_func in METRICS_INDEX.items():
            if method_name not in ["topk","roc_auc"]:
                result[method_name] = method_func.compute(
                        references=t_true, predictions=table.pred.values, **methods[method_name]
                    )
            else:
                result[method_name] = method_func.compute(
                        references=t_true, prediction_scores=table.loc[:,table.columns!="pred"].values, **methods[method_name]
                    )
        return result
    return compute_metrics

def load_tokenizer(tokenizer_path, *inputs, **kwargs):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, *inputs, **kwargs)
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer, 
        padding="max_length", 
        max_length=tokenizer.model_max_length
        )
    return tokenizer, data_collator

def load_config(
    tokenizer, model_configs_path, nlabels, token_dropout, positional_embedding_type, method
):
    if tokenizer is None:
        raise NotImplementedError("tokenizer has not load already.")
    model_configs=AutoConfig.from_pretrained(
            model_configs_path,
            vocab_size=len(tokenizer),
            pad_token_id=tokenizer.pad_token_id,
            mask_token_id=tokenizer.mask_token_id,
            num_labels=nlabels,
            token_dropout=token_dropout,
            positional_embedding_type=positional_embedding_type,
            finetuning_method=method,
        )
    return model_configs

def load_pretrained(config, pretrain_state_path):
    if os.path.exists(pretrain_state_path):
        model=EsmForTokenClassificationFrozen.from_pretrained(
            pretrain_state_path,
            config=config#,
            #device_map='auto'
        )
    else:
        print("No valid pretrained Model path")
        model=EsmForTokenClassificationFrozen(config)
    return model

def load_metrics(methods,ignore_label=-100):
    def compute_metrics(p):
        logits, labels = p
        softpred=softmax(logits,axis=2)
        pred_label=np.argmax(softpred,axis=2).astype(np.int8)
        logits=softpred.reshape((softpred.shape[0]*softpred.shape[1],-1))
        table=pd.DataFrame(logits)
        table["pred"]=np.array(pred_label).flatten()
        table["true"]=np.array(labels).flatten()
        table=table[table["true"]!=ignore_label]
        print("finish flatten")
        result = {}
        counts=table.true.value_counts().to_dict()
        result["topk"]={"topk":{
            k:sum(
                (table.sort_values(by=k,ascending=False)[:v]).true==k
                )/v 
            for k,v in  counts.items()
            }}

        return result
    return compute_metrics

def load_trainer(config_path,is_train=True,**kwargs):
    with open(config_path) as f:
        args = json.load(f)
    for k,v in kwargs.items():
        args["train_args"][k]=v
    tokenizer, data_collator = load_tokenizer(**args["tokenizer_args"])
    model_config = load_config(tokenizer, **args["configs_args"])
    model = load_pretrained(model_config, **args["model_args"])
    trainsets = load_data(tokenizer, **args["train_dataloader_args"])
    train_args = TrainingArguments(**args["train_args"])
    compute_metrics = load_metrics(**args["metric_args"])
    print(train_args)
    # 创建 TrainerCallback 实例
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=trainsets["train"],
        eval_dataset=trainsets["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    if not is_train :
        testsets = load_data(tokenizer, **args["test_dataloader_args"])
        return trainer,testsets["train"]
    else:
        return trainer
def train(config_path,*args,**kwargs):
    print(args,kwargs)
    trainer=load_trainer(config_path,**kwargs)
    trainer.train(resume_from_checkpoint=True)
if __name__=="__main__":
    import fire
    fire.Fire(train)