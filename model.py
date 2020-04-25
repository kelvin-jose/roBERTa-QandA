import config
import transformers
import torch
import torch.nn as nn


class Roberta(transformers.BertPreTrainedModel):
    def __init__(self, conf):
        super().__init__(conf)
        self.roberta = transformers.RobertaModel.from_pretrained(config.ROBERTA_PATH, config=conf)
        self.drop_out = nn.Dropout(0.1)
        self.out = nn.Linear(768 * 2, 2)
        self.init_weights()

    def forward(self, ids,  masks, token_type_ids):
        _, _, hidden_state = self.roberta(ids, attention_mask=masks, token_type_ids=token_type_ids)
        out = torch.cat((hidden_state[-1], hidden_state[-2]), dim=-1)
        out = self.drop_out(out)
        out = self.out(out)
        start, end = out.split(1, dim=-1)
        start, end = start.squeeze(-1), end.squeeze(-1)
        return start, end