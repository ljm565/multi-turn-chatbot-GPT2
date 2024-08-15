import gc
import time
import math
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch import distributed as dist

from tools.tokenizers import *
from tools import TrainingLogger, Evaluator, EarlyStopper
from trainer.build import get_model, get_data_loader, get_tokenizers
from utils import RANK, LOGGER, SCHEDULER_MSG, SCHEDULER_TYPE, colorstr, init_seeds
from utils.func_utils import *
from utils.filesys_utils import *
from utils.training_utils import *




class Trainer:
    def __init__(
            self, 
            config,
            mode: str,
            device,
            is_ddp=False,
            resume_path=None,
        ):
        init_seeds(config.seed + 1 + RANK, config.deterministic)

        # init
        self.mode = mode
        self.is_training_mode = self.mode in ['train', 'resume']
        self.device = torch.device(device)
        self.is_ddp = is_ddp
        self.is_rank_zero = True if not self.is_ddp or (self.is_ddp and device == 0) else False
        self.config = config
        self.scheduler_type = self.config.scheduler_type
        self.world_size = len(self.config.device) if self.is_ddp else 1
        if self.is_training_mode:
            self.save_dir = make_project_dir(self.config, self.is_rank_zero)
            self.wdir = self.save_dir / 'weights'

        # path, data params
        self.config.is_rank_zero = self.is_rank_zero
        self.resume_path = resume_path
        self.max_len = self.config.max_len
        self.metrics = self.config.metrics

        assert self.scheduler_type in SCHEDULER_TYPE, \
            SCHEDULER_MSG + f' but got {colorstr(self.scheduler_type)}'

        # init tokenizer, model, dataset, dataloader, etc.
        self.modes = ['train', 'validation'] if self.is_training_mode else ['train', 'validation', 'test']
        self.tokenizer = get_tokenizers(self.config, self.is_ddp)
        self.dataloaders = get_data_loader(self.config, self.tokenizer, self.modes, self.is_ddp)
        self.model = self._init_model(self.config, self.tokenizer, self.mode)
        self.training_logger = TrainingLogger(self.config, self.is_training_mode)
        self.evaluator = Evaluator(self.tokenizer)
        self.stopper, self.stop = EarlyStopper(self.config.patience), False

        # save the yaml config
        if self.is_rank_zero and self.is_training_mode:
            self.wdir.mkdir(parents=True, exist_ok=True)  # make dir
            self.config.save_dir = str(self.save_dir)
            yaml_save(self.save_dir / 'args.yaml', self.config)  # save run args
        
        # init criterion, optimizer, etc.
        self.steps = self.config.steps
        self.lr0 = self.config.lr0
        self.lrf = self.config.lrf
        self.epochs = math.ceil(self.steps / len(self.dataloaders['train'])) if self.is_training_mode else 1
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        if self.is_training_mode:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr0)

            # init scheduler
            self.warmup_steps_n = max(0, self.config.warmup_steps)
            if self.scheduler_type == 'cosine':
                self.lf = one_cycle(1, self.lrf, self.steps)
            elif self.scheduler_type == 'linear':
                self.lf = lambda x: (1 - (x - self.warmup_steps_n) / (self.steps - self.warmup_steps_n)) * (1.0 - self.lrf) + self.lrf
            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)
            if self.is_rank_zero:
                draw_training_lr_curve(self.config, self.lf, self.steps, self.warmup_steps_n, self.is_ddp, self.world_size)


    def _init_model(self, config, tokenizer, mode):
        def _resume_model(resume_path, device, is_rank_zero):
            try:
                checkpoints = torch.load(resume_path, map_location=device)
            except RuntimeError:
                LOGGER.warning(colorstr('yellow', 'cannot be loaded to MPS, loaded to CPU'))
                checkpoints = torch.load(resume_path, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoints['model'])
            del checkpoints
            torch.cuda.empty_cache()
            gc.collect()
            if is_rank_zero:
                LOGGER.info(f'Resumed model: {colorstr(resume_path)}')
            return model

        # init models
        do_resume = mode == 'resume' or (mode == 'validation' and self.resume_path)
        model = get_model(config, tokenizer, self.device)

        # resume model
        if do_resume:
            model = _resume_model(self.resume_path, self.device, config.is_rank_zero)

        # init ddp
        if self.is_ddp:
            torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.device])
        
        return model


    def do_train(self):
        self.train_cur_step = -1
        self.train_time_start = time.time()
        
        if self.is_rank_zero:
            LOGGER.info(f'\nUsing {self.dataloaders["train"].num_workers * (self.world_size or 1)} dataloader workers\n'
                        f"Logging results to {colorstr('bold', self.save_dir)}\n"
                        f'Starting training for {self.epochs} epochs...\n')
        
        if self.is_ddp:
            dist.barrier()

        for epoch in range(self.epochs):
            start = time.time()
            self.epoch = epoch

            if self.is_rank_zero:
                LOGGER.info('-'*100)

            for phase in self.modes:
                if self.is_rank_zero:
                    LOGGER.info('Phase: {}'.format(phase))

                if phase == 'train':
                    self.epoch_train(phase, epoch)
                    if self.is_ddp:
                        dist.barrier()
                else:
                    self.epoch_validate(phase, epoch)
                    if self.is_ddp:
                        dist.barrier()
            
            # clears GPU vRAM at end of epoch, can help with out of memory errors
            torch.cuda.empty_cache()
            gc.collect()

            # Early Stopping
            if self.is_ddp:  # if DDP training
                broadcast_list = [self.stop if self.is_rank_zero else None]
                dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                if not self.is_rank_zero:
                    self.stop = broadcast_list[0]
            
            if self.stop:
                break  # must break all DDP ranks

            if self.is_rank_zero:
                LOGGER.info(f"\nepoch {epoch+1} time: {time.time() - start} s\n\n\n")

        if RANK in (-1, 0) and self.is_rank_zero:
            LOGGER.info(f'\n{epoch + 1} epochs completed in '
                        f'{(time.time() - self.train_time_start) / 3600:.3f} hours.')
            

    def epoch_train(
            self,
            phase: str,
            epoch: int
        ):
        self.model.train()
        train_loader = self.dataloaders[phase]
        nb = len(train_loader)

        if self.is_ddp:
            train_loader.sampler.set_epoch(epoch)

        # init progress bar
        if RANK in (-1, 0):
            logging_header = ['CE Loss', 'lr']
            pbar = init_progress_bar(train_loader, self.is_rank_zero, logging_header, nb)

        for i, (src, trg) in pbar:
            # Warmup
            self.train_cur_step += 1
            if self.train_cur_step <= self.warmup_steps_n:
                self.optimizer.param_groups[0]['lr'] = lr_warmup(self.train_cur_step, self.warmup_steps_n, self.lr0, self.lf)
            cur_lr = self.optimizer.param_groups[0]['lr']
            
            batch_size = src.size(0)
            src, trg = src.to(self.device), trg.to(self.device)
            
            self.optimizer.zero_grad()
            _, output = self.model(src, trg)
            loss = self.criterion(output[:, :-1, :].reshape(-1, output.size(-1)), trg[:, 1:].reshape(-1))
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            if self.is_rank_zero:
                self.training_logger.update(
                    phase, 
                    epoch + 1,
                    self.train_cur_step,
                    batch_size, 
                    **{'train_loss': loss.item(), 'lr': cur_lr},
                )
                loss_log = [loss.item(), cur_lr]
                msg = tuple([f'{epoch + 1}/{self.epochs}'] + loss_log)
                pbar.set_description(('%15s' * 1 + '%15.4g' * len(loss_log)) % msg)
            
        # upadate logs
        if self.is_rank_zero:
            self.training_logger.update_phase_end(phase, printing=True)
        
        
    def epoch_validate(
            self,
            phase: str,
            epoch: int,
            is_training_now=True
        ):
        def _init_log_data_for_vis():
            data4vis = {'trg': [], 'pred': []}
            return data4vis

        def _append_data_for_vis(**kwargs):
            for k, v in kwargs.items():
                if isinstance(v, list):
                    self.data4vis[k].extend(v)
                else: 
                    self.data4vis[k].append(v)

        with torch.no_grad():
            if self.is_rank_zero:
                if not is_training_now:
                    self.data4vis = _init_log_data_for_vis()

                val_loader = self.dataloaders[phase]
                nb = len(val_loader)
                logging_header = ['CE Loss'] + self.config.metrics
                pbar = init_progress_bar(val_loader, self.is_rank_zero, logging_header, nb)

                self.model.eval()

                for i, (src, trg) in pbar:
                    batch_size = src.size(0)
                    src, trg = src.to(self.device), trg.to(self.device)

                    sources = [self.tokenizer.decode(s.tolist()) for s in src]
                    targets4metrics = [self.tokenizer.decode(t[1:].tolist()) for t in trg]

                    predictions, loss = self.model.batch_inference(
                        src=src,
                        start_tokens=trg[:, 0], 
                        max_len=self.max_len,
                        tokenizer=self.tokenizer,
                        loss_func=self.criterion,
                        target=trg
                    )
                
                    metric_results = self.metric_evaluation(loss, predictions, targets4metrics)

                    self.training_logger.update(
                        phase, 
                        epoch, 
                        self.train_cur_step if is_training_now else 0, 
                        batch_size, 
                        **{'validation_loss': loss.item()},
                        **metric_results
                    )

                    # logging
                    loss_log = [loss.item()]
                    msg = tuple([f'{epoch+1}/{self.epochs}'] + loss_log + [metric_results[k] for k in self.metrics])
                    pbar.set_description(('%15s' + '%15.4g' * (len(loss_log) + len(self.metrics))) % msg)

                    ids = random.sample(range(batch_size), min(self.config.prediction_print_n, batch_size))
                    for id in ids:
                        print_samples(' '.join(sources[id].split()[1:]), targets4metrics[id], predictions[id])

                    if not is_training_now:
                        _append_data_for_vis(
                            **{'trg': targets4metrics,
                               'pred': predictions}
                        )

                # upadate logs and save model
                self.training_logger.update_phase_end(phase, printing=True)
                if is_training_now:
                    self.training_logger.save_model(self.wdir, self.model)
                    self.training_logger.save_logs(self.save_dir)

                    high_fitness = self.training_logger.model_manager.best_higher
                    low_fitness = self.training_logger.model_manager.best_lower
                    self.stop = self.stopper(epoch + 1, high=high_fitness, low=low_fitness)

    
    def metric_evaluation(self, loss, response_pred, response_gt):
        metric_results = {k: 0 for k in self.metrics}
        for m in self.metrics:
            if m == 'ppl':
                metric_results[m] = self.evaluator.cal_ppl(loss.item())
            elif m == 'bleu2':
                metric_results[m] = self.evaluator.cal_bleu_score(response_pred, response_gt, n=2)
            elif m == 'bleu4':
                metric_results[m] = self.evaluator.cal_bleu_score(response_pred, response_gt, n=4)
            elif m == 'nist2':
                metric_results[m] = self.evaluator.cal_nist_score(response_pred, response_gt, n=2)
            elif m == 'nist4':
                metric_results[m] = self.evaluator.cal_nist_score(response_pred, response_gt, n=4)
            else:
                LOGGER.warning(f'{colorstr("red", "Invalid key")}: {m}')
        
        return metric_results
    
    
    def chatting(self, query):
        query = [self.tokenizer.bos_token_id] + self.tokenizer.encode(query)[:self.max_len-2] + [self.tokenizer.eos_token_id]
        query = query + [self.tokenizer.pad_token_id] * (self.max_len - len(query))
        
        query = torch.LongTensor(query).unsqueeze(0).to(self.device)
        trg = torch.LongTensor([self.tokenizer.bos_token_id]).unsqueeze(0).to(self.device)

        decoder_all_output = []
        for i in range(self.max_len):
            if i == 0:
                trg = trg[:, i].unsqueeze(1)
                _, output = self.model(query, trg)
                trg = torch.cat((trg, torch.argmax(output[:, -1], dim=-1).unsqueeze(1)), dim=1)
            else:
                _, output = self.model(query, trg)
                trg = torch.cat((trg, torch.argmax(output[:, -1], dim=-1).unsqueeze(1)), dim=1)

            decoder_all_output.append(output[:, -1].unsqueeze(1).detach().cpu())
        decoder_all_output = torch.argmax(torch.cat(decoder_all_output, dim=1), dim=-1)
        decoder_all_output = self.tokenizer.decode(decoder_all_output[0].tolist())
        
        return decoder_all_output
