# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Accuracy metric."""

import datasets
import numpy as np
import pandas as pd
import numpy as np
import os
from scipy.special import softmax
from sklearn.metrics import precision_recall_curve, auc

import evaluate


_DESCRIPTION = """
This metric computes the area under the curve (AUC) for the Receiver Operating Characteristic Curve (ROC). The return values represent how well the model used is predicting the correct classes, based on the input data. A score of `0.5` means that the model is predicting exactly at chance, i.e. the model's predictions are correct at the same rate as if the predictions were being decided by the flip of a fair coin or the roll of a fair die. A score above `0.5` indicates that the model is doing better than chance, while a score below `0.5` indicates that the model is doing worse than chance.

This metric has three separate use cases:
    - binary: The case in which there are only two different label classes, and each example gets only one label. This is the default implementation.
    - multiclass: The case in which there can be more than two different label classes, but each example still gets only one label.
    - multilabel: The case in which there can be more than two different label classes, and each example can have more than one label.
"""

_KWARGS_DESCRIPTION = """PRAUC"""

_CITATION = """\
@article{doi:10.1177/0272989X8900900307,
author = {Donna Katzman McClish},
title ={Analyzing a Portion of the ROC Curve},
journal = {Medical Decision Making},
volume = {9},
number = {3},
pages = {190-195},
year = {1989},
doi = {10.1177/0272989X8900900307},
    note ={PMID: 2668680},
URL = {https://doi.org/10.1177/0272989X8900900307},
eprint = {https://doi.org/10.1177/0272989X8900900307}
}


@article{10.1023/A:1010920819831,
author = {Hand, David J. and Till, Robert J.},
title = {A Simple Generalisation of the Area Under the ROC Curve for Multiple Class Classification Problems},
year = {2001},
issue_date = {November 2001},
publisher = {Kluwer Academic Publishers},
address = {USA},
volume = {45},
number = {2},
issn = {0885-6125},
url = {https://doi.org/10.1023/A:1010920819831},
doi = {10.1023/A:1010920819831},
journal = {Mach. Learn.},
month = {oct},
pages = {171–186},
numpages = {16},
keywords = {Gini index, AUC, error rate, ROC curve, receiver operating characteristic}
}


@article{scikit-learn,
title={Scikit-learn: Machine Learning in {P}ython},
author={Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V. and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P. and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E.},
journal={Journal of Machine Learning Research},
volume={12},
pages={2825--2830},
year={2011}
}
"""
# 

def compute_binary_pr_auc(reference, predict_logits):
    precision, recall, _ = precision_recall_curve(reference, predict_logits)
    pr_auc = auc(recall, precision)
    return pr_auc

def compute_ovr_pr_auc(reference, predict_logits, average=None,ignore_idx=[]):
    n_classes = predict_logits.shape[1]
    pr_aucs = []
    for class_idx in range(n_classes):
        if class_idx not in ignore_idx:
            pr_auc = compute_binary_pr_auc((reference == class_idx).astype(int), predict_logits[:, class_idx])
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


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class PRAUC(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "prediction_scores": datasets.Sequence(datasets.Value("float")),
                    "references": datasets.Value("int32"),
                }
                if self.config_name == "multiclass"
                else {
                    "references": datasets.Sequence(datasets.Value("int32")),
                    "prediction_scores": datasets.Sequence(datasets.Value("float")),
                }
                if self.config_name == "multilabel"
                else {
                    "references": datasets.Value("int32"),
                    "prediction_scores": datasets.Value("float"),
                }
            ),
            reference_urls=["https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html"],
        )

    def _compute(
        self,
        references,
        prediction_scores,
        average="macro",
        multi_class="raise",
    ):
        return {
            "pr_auc": pr_auc_score(
                np.array(references), np.array(prediction_scores), multi_class, average
            )
        }

