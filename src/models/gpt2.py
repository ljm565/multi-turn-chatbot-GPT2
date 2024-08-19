from transformers import GPT2LMHeadModel

import torch
import torch.nn as nn

from utils import LOGGER, colorstr



class GPT2(nn.Module):
    def __init__(self, config, tokenizer):
        super(GPT2, self).__init__()
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
    

    def batch_inference(self, src, start_tokens, max_len, tokenizer, loss_func=None, target=None):
        loss = None
        if loss_func:
            assert target != None, LOGGER(colorstr('red', 'Target must be required if you want to return loss values..'))
            output = self.forward(src)
            loss = loss_func(output[:, :-1, :].reshape(-1, output.size(-1)), target[:, 1:].reshape(-1))
        
        if isinstance(start_tokens, tuple):
            st, stl = start_tokens
            start_tokens = [single_s[:single_sl].unsqueeze(0) for single_s, single_sl in zip(st, stl)]
        else:
            start_tokens = [start_tokens.unsqueeze(1)]
        
        # Due to decoder-only architecture, token length of every single batch is different
        preds = []
        for start_token in start_tokens:
            while start_token.size(1) < max_len:
                output = self.forward(start_token)
                start_token = torch.cat((start_token, torch.argmax(output[:, -1], dim=-1).unsqueeze(1)), dim=1)
            preds.append(start_token[0])

        predictions = [tokenizer.decode(pred.detach().cpu().tolist()) for pred in preds]

        return predictions, loss