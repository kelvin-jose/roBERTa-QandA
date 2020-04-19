import config
import transformers
import torch.nn as nn


class Roberta(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.roberta = transformers.RobertaModel.from_pretrained(config.ROBERTA_PATH, config=conf)
        self.out = nn.Linear(768, 2)

    def forward(self, ids,  masks, token_type_ids):
        sequence, pooler = self.roberta(ids, attention_mask=masks, token_type_ids=token_type_ids)
        out = self.out(sequence)
        start, end = out.split(1, dim=-1)
        start, end = start.squeeze(-1), end.squeeze(-1)
        return start, end