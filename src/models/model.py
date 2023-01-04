import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel


class GPTChatbot(nn.Module):
    def __init__(self, config, tokenizer):
        super(GPTChatbot, self).__init__()
        self.pretrained_model = config.pretrained_model
        self.model = GPT2LMHeadModel.from_pretrained(self.pretrained_model)
        self.model.resize_token_embeddings(config.vocab_size)
        self.pad_token_id = tokenizer.pad_token_id


    def make_mask(self, x):
        pad_mask = torch.where(x==self.pad_token_id, 0, 1)
        return pad_mask


    def forward(self, x):
        pad_mask = self.make_mask(x)
        output = self.model(input_ids=x, attention_mask=pad_mask)
        return output.logits