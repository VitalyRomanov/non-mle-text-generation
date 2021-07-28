import transformers
from torch import nn


class BleurtModel(nn.Module):
    """
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bert = transformers.BertModel(config)
        self.dense = nn.Linear(config.hidden_size,1)

    def forward(self, input_ids, input_mask, segment_ids):
        cls_state = self.bert(input_ids, input_mask,
                              segment_ids).pooler_output
        return self.dense(cls_state)