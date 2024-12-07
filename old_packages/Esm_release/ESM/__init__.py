import json
import os

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, load_from_disk
from scipy.special import softmax
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from .EsmForTokenClassification import EsmForTokenClassification
from .metrics import METRICS_INDEX


def group_texts(tokenizer):
    def _inner(data):
        tokenized_inputs = tokenizer(
            data["seq"], truncation=True, max_length=tokenizer.model_max_length
        )
        return tokenized_inputs

    return _inner


def load_data(
    tokenizer,
    set_path=None,
    preload_data=None,
    seed=None,
    test_size=None,
):
    if os.path.exists(preload_data):
        data_for_preload = load_from_disk(preload_data)
    else:
        dataset = load_dataset("json", data_files=set_path)
        data_for_preload = dataset.map(
            group_texts(tokenizer), batched=True, num_proc=os.cpu_count()
        )
        data_for_preload = data_for_preload.rename_column("label", "labels")
        data_for_preload = data_for_preload.remove_columns("seq")
        if not test_size is None:
            data_for_preload = data_for_preload["train"].train_test_split(
                test_size=test_size, shuffle=True, seed=seed
            )
        data_for_preload.save_to_disk(preload_data, num_proc=os.cpu_count())
    return data_for_preload


def load_tokenizer(tokenizer_path, *inputs, **kwargs):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, *inputs, **kwargs)
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer, padding="max_length", max_length=tokenizer.model_max_length
    )
    return tokenizer, data_collator


def load_config(
    tokenizer,
    model_configs_path,
    nlabels,
    token_dropout,
    positional_embedding_type,
    method,
):
    if tokenizer is None:
        raise NotImplementedError("tokenizer has not load already.")
    model_configs = AutoConfig.from_pretrained(
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


def load_pretrained(config, pretrain_state_path, pretrain_freeze=False):
    if os.path.exists(pretrain_state_path):
        model = EsmForTokenClassification.from_pretrained(
            pretrain_state_path,
            config=config,  # ,
            # device_map='auto'
        )
    else:
        print("No valid pretrained Model path")
        model = EsmForTokenClassification(config)
    if pretrain_freeze:
        print("ESM freeze")
        for p in model.esm.parameters():
            p.requires_grad = False
        print(p)
    return model


def compute_binary_pr_auc(reference, predict_logits):
    precision, recall, _ = precision_recall_curve(reference, predict_logits)
    return auc(recall, precision)


def compute_ovr_pr_auc(reference, predict_logits, average=None, ignore_idx=[]):
    n_classes = predict_logits.shape[1]
    pr_aucs = []
    for class_idx in range(n_classes):
        if class_idx not in ignore_idx:
            pr_auc = compute_binary_pr_auc(
                (reference == class_idx).astype(int), predict_logits[:, class_idx]
            )
            pr_aucs.append(pr_auc)
    if average == "macro":
        return np.mean(pr_aucs)
    elif average == "weighted":
        class_counts = np.bincount(reference)
        weighted_pr_aucs = np.array(pr_aucs) * class_counts / len(reference)
        return np.sum(weighted_pr_aucs)
    else:
        return pr_aucs


def compute_ovo_pr_auc(reference, predict_logits, average=None):
    # OvO is not directly supported by precision_recall_curve
    raise NotImplementedError("OvO PR AUC computation is not implemented yet.")


def pr_auc_score(reference, predict_logits, multi_class=None, average=None):
    if multi_class == "ovr":
        pr_auc = compute_ovr_pr_auc(reference, predict_logits, average=average)
    elif multi_class == "ovo":
        pr_auc = compute_ovo_pr_auc(reference, predict_logits, average=average)
    else:
        pr_auc = compute_binary_pr_auc(reference, predict_logits)
    return pr_auc


def load_metrics(ignore_label=-100):
    def compute_metrics(p):
        logits, labels = p
        softpred = softmax(logits, axis=2)
        pred_label = np.argmax(softpred, axis=2).astype(np.int8)
        logits = softpred.reshape((softpred.shape[0] * softpred.shape[1], -1))
        table = pd.DataFrame(logits)
        table["pred"] = np.array(pred_label).flatten()
        table["true"] = np.array(labels).flatten()
        table = table[table["true"] != ignore_label]
        print("finish flatten")
        result = {}
        counts = table.true.value_counts().to_dict()
        result["topk"] = {
            "topk": {
                k: sum((table.sort_values(by=k, ascending=False)[:v]).true == k) / v
                for k, v in counts.items()
            }
        }
        scores = table.loc[
            :, table.columns[~table.columns.isin(["pred", "true"])]
        ].values
        result["roc_auc"] = list(
            roc_auc_score(table["true"], scores, multi_class="ovr", average=None)
        )
        result["pr_auc"] = list(
            pr_auc_score(table["true"], scores, multi_class="ovr", average=None)
        )
        return result

    return compute_metrics


def load_trainer(config_path, is_train=True, **kwargs):
    with open(config_path) as f:
        args = json.load(f)
    for k, v in kwargs.items():
        args["train_args"][k] = v
    tokenizer, data_collator = load_tokenizer(**args["tokenizer_args"])
    model_config = load_config(tokenizer, **args["configs_args"])
    model = load_pretrained(model_config, **args["model_args"])
    trainsets = load_data(tokenizer, **args["train_dataloader_args"])
    train_args = TrainingArguments(**args["train_args"])
    compute_metrics = load_metrics()
    print(train_args)
    # 创建 TrainerCallback 实例
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=trainsets["train"],
        eval_dataset=trainsets["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    if not is_train:
        testsets = load_data(tokenizer, **args["test_dataloader_args"])
        return trainer, testsets["train"]
    else:
        return trainer
