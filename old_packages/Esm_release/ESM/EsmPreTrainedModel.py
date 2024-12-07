from .EsmConfig import EsmConfig
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers.modeling_utils import (
    PreTrainedModel,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)

class EsmPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = EsmConfig
    base_model_prefix = "esm"
    supports_gradient_checkpointing = True
    _no_split_modules = [
        "EsmLayer",
        "EsmFoldTriangularSelfAttentionBlock",
        "EsmEmbeddings",
    ]

    # Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
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

