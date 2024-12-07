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
"""Recall metric."""

import datasets
import numpy as np

import evaluate

_DESCRIPTION = """
TOPK....
"""
_KWARGS_DESCRIPTION = """
TOPK....
"""
_CITATION = """
@article{scikit-learn, title={Scikit-learn: Machine Learning in {P}ython}, author={Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V. and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P. and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E.}, journal={Journal of Machine Learning Research}, volume={12}, pages={2825--2830}, year={2011}
"""

def top_k_type(pred_score, true_label, class_value, k, count=None,slack=0,slack_ratio=1):
    sort_index = (-1 * pred_score).argsort()#index
    sort_label = true_label[sort_index]
    sort_logtis = pred_score[sort_index]
    soft_k=min(int(slack_ratio*k+slack)+1,len(sort_label))
    if count:
        return sort_logtis[:soft_k],sort_label[:soft_k],k
    else:
        true_label=sum(sort_label==class_value)
        accuracy = true_label / soft_k
        return accuracy

@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class TopK(evaluate.Metric):
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
            reference_urls=["https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html"],
        )

    def _compute(
        self,
        references,
        prediction_scores,
        count=None,
        ignore_label=[],
        slack=0,
        slack_ratio=1
    ):
        references,prediction_scores=np.array(references),np.array(prediction_scores)
        class_values, class_ks = np.unique(references, return_counts=True)
        ans = {
            class_values[i]: top_k_type(
                pred_score=prediction_scores[:,i],
                true_label=references,
                class_value=class_values[i],
                k=class_ks[i],
                count=count,
                slack=slack,
                slack_ratio=slack_ratio
            )
            for i in range(len(class_values))
            if i not in ignore_label
        }
        
        return {"topk": ans}
