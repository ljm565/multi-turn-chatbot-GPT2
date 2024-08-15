import os
import subprocess

import torch
from torch import distributed as dist
from torch.utils.data import DataLoader, distributed

from models import GPT2
from tools.tokenizers import CustomGPT2Tokenizer
from utils import LOGGER, RANK, colorstr
from utils.filesys_utils import read_dataset
from utils.data_utils import DLoader, seed_worker

PIN_MEMORY = str(os.getenv('PIN_MEMORY', True)).lower() == 'true'  # global pin_memory for dataloaders



def get_tokenizers(config):
    if config.dailydialog_train:
        tokenizer = CustomGPT2Tokenizer(config)
        config.vocab_size = tokenizer.vocab_size
    else:
        LOGGER.warning(colorstr('yellow', 'You must implement your custom tokenizer loading codes..'))
        raise NotImplementedError
    return tokenizer


def get_model(config, tokenizer, device):
    model = GPT2(config, tokenizer).to(device)
    return model


def build_dataset(config, tokenizer, modes):
    if config.dailydialog_train:
        dataset_dir = os.path.join(config.dailydialog_dataset.path, 'dailydialog/processed')
        dataset_paths = {mode: os.path.join(dataset_dir, f'dailydialog.{mode}') if mode != 'validation' \
                                                else os.path.join(dataset_dir, 'dailydialog.val') for mode in modes}
        dataset_dict = {s: DLoader(read_dataset(p), tokenizer, config) for s, p in dataset_paths.items() if s in dataset_paths}
    else:
        LOGGER.warning(colorstr('yellow', 'You have to implement data pre-processing code..'))
        # dataset_dict = {mode: CustomDLoader(config.CUSTOM.get(f'{mode}_data_path')) for mode in modes}
        raise NotImplementedError
    return dataset_dict


def build_dataloader(dataset, batch, workers, shuffle=True, is_ddp=False):
    batch = min(batch, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch if batch > 1 else 0, workers])  # number of workers
    sampler = None if not is_ddp else distributed.DistributedSampler(dataset, shuffle=shuffle)
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK)
    return DataLoader(dataset=dataset,
                              batch_size=batch,
                              shuffle=shuffle and sampler is None,
                              num_workers=nw,
                              sampler=sampler,
                              pin_memory=PIN_MEMORY,
                              collate_fn=getattr(dataset, 'collate_fn', None),
                              worker_init_fn=seed_worker,
                              generator=generator)


def get_data_loader(config, tokenizer, modes, is_ddp=False):
    datasets = build_dataset(config, tokenizer, modes)
    dataloaders = {m: build_dataloader(datasets[m], 
                                       config.batch_size, 
                                       config.workers, 
                                       shuffle=(m == 'train'), 
                                       is_ddp=is_ddp) for m in modes}

    return dataloaders