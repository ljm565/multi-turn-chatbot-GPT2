import os
import math
import numpy as np
import matplotlib.pyplot as plt

from utils import LOGGER, TQDM


def one_cycle(y1=0.0, y2=1.0, steps=100):
    """Returns a lambda function for sinusoidal ramp from y1 to y2 https://arxiv.org/pdf/1812.01187.pdf."""
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1


def init_progress_bar(dloader, is_rank_zero, loss_names, nb):
    if is_rank_zero:
        header = tuple(['Epoch'] + loss_names)
        LOGGER.info(('\n' + '%15s' * (1 + len(loss_names))) % header)
        pbar = TQDM(enumerate(dloader), total=nb)
    else:
        pbar = enumerate(dloader)
    return pbar


def choose_proper_resume_model(resume_dir, type):
    weights_dir = os.listdir(os.path.join(resume_dir, 'weights'))
    try:
        weight = list(filter(lambda x: type in x, weights_dir))[0]
        return os.path.join(resume_dir, 'weights', weight)
    except IndexError:
        raise IndexError(f"There's no model path in {weights_dir} of type {type}")
    

def draw_training_lr_curve(config, func, all_steps_n, warmup_steps_n, is_ddp, world_size):
    save_dir = os.path.join(config.save_dir, 'vis_outputs')
    lr0 = config.lr0
    
    os.makedirs(save_dir, exist_ok=True)
    lrs = [func(i)*lr0 if i > warmup_steps_n
           else lr_warmup(i, warmup_steps_n, lr0, func) for i in range(all_steps_n)]
    plt.figure(figsize=(8, 6))
    plt.plot(range(all_steps_n), lrs, marker='o')
    plt.xlabel('Steps')
    plt.ylabel('Learning Rate')
    if is_ddp:
        plt.title(f'Learning Rate Schedule per GPU (World Size: {world_size})')
    else:
        plt.title('Learning Rate Schedule')
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'lr_schedule.png'))


def lr_warmup(cur_step, warmup_steps_n, lr0, func):
    new_lr = np.interp(
        cur_step, 
        [0, warmup_steps_n], 
        [0, lr0*func(cur_step)]
    )
    return new_lr