import config
import transformers
import torch.nn as nn


class BERTBaseUncased(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = transformers.BertModel.from_pretrained(
            config.BERT_PATH
        )
        self.bert_dropout = nn.Dropout(0.3)
        self.output = nn.Linear(768, config.NUM_CLASSES) # the bert base model has 768 output features

    def forward(self, ids, mask, token_type_ids):
        _, output2 = self.bert(
            ids, attention_mask=mask,
            token_type_ids=token_type_ids
            )

        bert_output = self.bert_dropout(output2)
        output = self.output(bert_output)
        return output 

    