
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers.modeling_outputs import TokenClassifierOutput

from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)
from transformers.utils import logging

from .EsmModel import (
    EsmPreTrainedModel,
    EsmModel,
    ESM_START_DOCSTRING,
    ESM_INPUTS_DOCSTRING,
    _CHECKPOINT_FOR_DOC,
    _CONFIG_FOR_DOC,
)

logger = logging.get_logger(__name__)

@add_start_docstrings(
    """
    ESM Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    ESM_START_DOCSTRING,
)
class EsmForTokenClassification(EsmPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.esm = EsmModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.hidden =  nn.Linear(config.hidden_size,config.observation)
        # self.classifier = nn.Linear(config.observation,config.num_labels)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()#TODO: Compare with self.apply(init_weights) (which redefines the function `init_weights`)

    @add_start_docstrings_to_model_forward(
        ESM_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
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
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

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

        sequence_output = outputs[0]#(batchsize,Length,HiddenSize)

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()#ignoring label:-100

            labels = labels.to(logits.device)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        if output_attentions == True:
            return TokenClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=None,#outputs.hidden_states,
                attentions=outputs.attentions,
            )
        else:
            return TokenClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=None#outputs.hidden_states,
            )
            

