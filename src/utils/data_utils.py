import random
import numpy as np

import torch
from torch.utils.data import Dataset



def seed_worker(worker_id):  # noqa
    """Set dataloader worker seed https://pytorch.org/docs/stable/notes/randomness.html#dataloader."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class DLoader(Dataset):
    def __init__(self, data, tokenizer, config):
        self.data = data
        self.tokenizer = tokenizer
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.max_len = config.max_len
        self.length = len(self.data)


    def make_data(self, data_list):
        total_s = [self.cls_token_id]
        for s in data_list:
            tmp_s = self.tokenizer.encode(s) + [self.sep_token_id]
            if len(total_s) + len(tmp_s) > self.max_len:
                break
            total_s += tmp_s
        total_s += [self.pad_token_id] * (self.max_len - len(total_s))
        return total_s


    def __getitem__(self, idx):
        s = self.make_data(self.data[idx])
        return torch.LongTensor(s)


    def __len__(self):
        return self.length