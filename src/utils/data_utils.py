import random
import numpy as np
from copy import deepcopy

import torch
from torch.utils.data import Dataset

from utils import colorstr



def seed_worker(worker_id):  # noqa
    """Set dataloader worker seed https://pytorch.org/docs/stable/notes/randomness.html#dataloader."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class DLoader(Dataset):
    def __init__(self, data, tokenizer, config):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = config.max_len
        self.length = len(self.data)


    def make_data(self, multi_turn_sentences):
        all_turns_tokens = [self.tokenizer.cls_token_id]
        label_tokens = [self.tokenizer.pad_token_id]

        for i, sentence in enumerate(multi_turn_sentences):
            # source tokens
            sentence_token = self.tokenizer.encode(sentence) + [self.tokenizer.sep_token_id]
            all_turns_tokens.extend(sentence_token)

            if i == 0:
                first_sentence = self.padding(deepcopy(all_turns_tokens)[:self.max_len], self.max_len, self.tokenizer.pad_token_id)
                first_sentence_l = len(all_turns_tokens)

            # label tokens
            if i % 2  == 1:
                label_tokens.extend(sentence_token)
            else:
                label_tokens.extend([self.tokenizer.pad_token_id] * len(sentence_token))
            
            if len(all_turns_tokens) >= self.max_len:
                break
        
        # Adding EOS token at the end when the length of tokens cannot be reached to the max length
        # even though all turns are appended
        if len(all_turns_tokens) < self.max_len:
            # source tokens
            all_turns_tokens += [self.tokenizer.eos_token_id]
            all_turns_tokens = self.padding(all_turns_tokens, self.max_len, self.tokenizer.pad_token_id)
            
            # label tokens
            label_tokens += [self.tokenizer.eos_token_id]
            label_tokens = self.padding(label_tokens, self.max_len, self.tokenizer.pad_token_id)
        else:
            all_turns_tokens = all_turns_tokens[:self.max_len]
            label_tokens = label_tokens[:self.max_len]

        assert len(all_turns_tokens) == len(label_tokens) == self.max_len, \
            colorstr(f'The token length must be equal to the max length of the config.yaml. Expected {self.max_len}, but got {len(all_turns_tokens)}')

        return all_turns_tokens, label_tokens, first_sentence, first_sentence_l
    

    @staticmethod
    def padding(x:list, length, pad_token_id):
        x += [pad_token_id] * (length - len(x))
        return x


    def __getitem__(self, idx):
        x, y, fs, fsl = self.make_data(self.data[idx])
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long), \
                torch.tensor(fs, dtype=torch.long), torch.tensor(fsl, dtype=torch.long)


    def __len__(self):
        return self.length