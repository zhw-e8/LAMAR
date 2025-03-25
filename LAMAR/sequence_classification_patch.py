import torch
from torch import nn
from typing import Optional, Tuple, Union
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F
from transformers.modeling_outputs import SequenceClassifierOutput
import math
from LAMAR.modeling_nucESM2 import EsmModel
from LAMAR.LossFunctions import FocalLoss


def gelu(x):
    """
    This is the gelu implementation from the original ESM repo. Using F.gelu yields subtly wrong results.
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    
    
class EsmForSequenceClassification(nn.Module):

    def __init__(self, config, head_type, freeze, kernel_sizes, ocs):
        super().__init__()
        self.num_labels = config.num_labels
        self.config = config
        self.freeze = freeze

        self.esm = EsmModel(config, add_pooling_layer=False)
        self.classifier = EsmClassificationHead(config, head_type, kernel_sizes, ocs)

        self.apply(self.init_weights)

    def init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv1d):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.freeze:
            with torch.no_grad():
                outputs = self.esm(
                    input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
        else:
            outputs = self.esm(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)

            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "single_label_classification_not_balance":
                loss_fct = FocalLoss()
                loss = loss_fct(logits.view(-1, self.num_labels)[:, 1], labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class EsmClassificationHead(nn.Module):

    def __init__(self, config, head_type, kernel_sizes, ocs):
        super().__init__()
        self.head_type = head_type
        if self.head_type == 'Linear':
            self.head = EsmSequenceClassificationLinearHead(config)
        elif self.head_type == 'CNN':
            self.head = EsmSequenceClassificationCNNHead(config, kernel_sizes, ocs)
        
    def forward(self, features):
        x = self.head(features)
        return x
    

class EsmSequenceClassificationLinearHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
    
    
class EsmSequenceClassificationCNNHead(nn.Module):

    def __init__(self, config, kernel_sizes, ocs):
        """
        kernel_sizes: such as [2, 3, 5]
        """
        super().__init__()
        self.convs = nn.ModuleList(
            [nn.Conv1d(in_channels=config.hidden_size, out_channels=ocs, kernel_size=kernel_size, stride=1, padding=0, dilation=1, bias=False, padding_mode='zeros') for kernel_size in kernel_sizes]
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(ocs * len(kernel_sizes), config.num_labels)

    def forward(self, features):
        x = features.transpose(1, 2) # features: [batch size, seq len, hidden size], x: [batch size, hidden size, seq len]
        x = [self.relu(conv(x)) for conv in self.convs] # x: [[batch size, out channels, seq len]]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] # x: [[batch size, out channels]]
        x = torch.cat(x, dim=1) # x: [batch size, out channels * len(self.convs)]
        x = self.out_proj(x)
        return x
    
    
