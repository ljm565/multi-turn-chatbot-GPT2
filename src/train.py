import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
import pickle
from tokenizer import Tokenizer
import random
import time
from tqdm import tqdm

from utils.config import Config
from utils.utils_func import *
from utils.utils_data import DLoader
from models.model import GPTChatbot



class Trainer:
    def __init__(self, config:Config, device:torch.device, mode:str, continuous:int):
        self.config = config
        self.device = device
        self.mode = mode
        self.continuous = continuous
        self.dataloaders = {}

        # if continuous, load previous training info
        if self.continuous:
            with open(self.config.loss_data_path, 'rb') as f:
                self.loss_data = pickle.load(f)

        # path, data params
        self.base_path = self.config.base_path
        self.model_path = self.config.model_path
        self.data_path = self.config.dataset_path
 
        # train params
        self.batch_size = self.config.batch_size
        self.epochs = self.config.epochs
        self.lr = self.config.lr
        self.max_len = self.config.max_len
        self.result_num = self.config.result_num

        # define tokenizer
        self.tokenizer = Tokenizer(self.config)
        self.config.vocab_size = self.tokenizer.vocab_size

        # dataloader
        torch.manual_seed(999)  # for reproducibility
        if self.mode == 'train':
            self.dataset = {s: DLoader(load_dataset(p), self.tokenizer, self.config) for s, p in self.data_path.items()}
            self.dataloaders = {
                s: DataLoader(d, self.batch_size, shuffle=True) if s == 'train' else DataLoader(d, self.batch_size, shuffle=False)
                for s, d in self.dataset.items()}
        else:
            self.dataset = {s: DLoader(load_dataset(p), self.tokenizer, self.config) for s, p in self.data_path.items() if s == 'test'}
            self.dataloaders = {s: DataLoader(d, self.batch_size, shuffle=False) for s, d in self.dataset.items() if s == 'test'}

        # model, optimizer, loss
        self.model = GPTChatbot(self.config, self.tokenizer).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
    
        if self.mode == 'train':
            total_steps = len(self.dataloaders['train']) * self.epochs
            pct_start = 100 / total_steps
            final_div_factor = self.lr / 25 / 2.5e-6    # OneCycleLR default value is 25
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            self.scheduler = OneCycleLR(self.optimizer, max_lr=self.lr, total_steps=total_steps, pct_start=pct_start, final_div_factor=final_div_factor)
            if self.continuous:
                self.check_point = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(self.check_point['model'])
                self.optimizer.load_state_dict(self.check_point['optimizer'])
                del self.check_point
                torch.cuda.empty_cache()
        else:
            self.check_point = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(self.check_point['model'])    
            self.model.eval()
            del self.check_point
            torch.cuda.empty_cache()

        
    def training(self):
        early_stop = 0
        best_val_bleu = 0 if not self.continuous else self.loss_data['best_val_bleu']
        train_loss_history = [] if not self.continuous else self.loss_data['train_loss_history']
        val_score_history = {'bleu2': [], 'bleu4': [], 'nist2': [], 'nist4': []} if not self.continuous else self.loss_data['val_score_history']
        best_epoch_info = 0 if not self.continuous else self.loss_data['best_epoch']

        for epoch in range(self.epochs):
            start = time.time()
            print(epoch+1, '/', self.epochs)
            print('-'*10)
            for phase in ['train', 'val', 'test']:
                print('Phase: {}'.format(phase))
                if phase == 'train':
                    epoch_loss = self.train(phase, epoch)
                    train_loss_history.append(epoch_loss)
                else:
                    bleu2, bleu4, nist2, nist4 = self.inference(phase, self.result_num)
                    if phase == 'val':
                        val_score_history['bleu2'].append(bleu2)
                        val_score_history['bleu4'].append(bleu4)
                        val_score_history['nist2'].append(nist2)
                        val_score_history['nist4'].append(nist4)

                        # save best model
                        early_stop += 1
                        if  val_score_history['bleu4'][-1] > best_val_bleu:
                            early_stop = 0
                            best_val_bleu = val_score_history['bleu4'][-1]
                            best_epoch = best_epoch_info + epoch + 1
                            save_checkpoint(self.model_path, self.model, self.optimizer)

            print("time: {} s\n".format(time.time() - start))
            print('\n'*2)

            # early stopping
            if early_stop == self.config.early_stop_criterion:
                break

        print('best val bleu: {:4f}, best epoch: {:d}\n'.format(best_val_bleu, best_epoch))
        self.loss_data = {'best_epoch': best_epoch, 'best_val_bleu': best_val_bleu, 'train_loss_history': train_loss_history, 'val_score_history': val_score_history}
        return self.loss_data


    def train(self, phase, epoch):
        self.model.train()
        total_loss = 0

        for i, x in enumerate(self.dataloaders[phase]):
            batch = x.size(0)
            x = x.to(self.device)
            self.optimizer.zero_grad()

            with torch.set_grad_enabled(phase=='train'):
                output = self.model(x)
                loss = self.criterion(output[:, :-1, :].reshape(-1, output.size(-1)), x[:, 1:].reshape(-1))
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

            total_loss += loss.item()*batch

            if i % 30 == 0:
                print('Epoch {}: {}/{} step loss: {}'.format(epoch+1, i, len(self.dataloaders[phase]), loss.item()))

        epoch_loss = total_loss/len(self.dataloaders[phase].dataset)

        print('{} loss: {:4f}\n'.format(phase, epoch_loss))
        return epoch_loss


    def inference(self, phase, result_num=3):
        self.model.eval()

        all_trg, all_output = [], []
        with torch.no_grad():
            for x in tqdm(self.dataloaders[phase], desc=phase + ' inferencing..'):
                x = x.to(self.device)
                output = self.model(x)
                
                x = x.detach().cpu()
                output = torch.argmax(output.detach().cpu(), dim=-1)
                all_trg.append(x)
                all_output.append(output)

        # calculate scores
        all_ref, all_pred = tensor2list(all_trg, all_output, self.tokenizer)
        bleu2 = cal_scores(all_ref, all_pred, 'bleu', 2)
        bleu4 = cal_scores(all_ref, all_pred, 'bleu', 4)
        nist2 = cal_scores(all_ref, all_pred, 'nist', 2)
        nist4 = cal_scores(all_ref, all_pred, 'nist', 4)
        print('\nInference Score')
        print('bleu2: {}, bleu4: {}, nist2: {}, nist4: {}'.format(bleu2, bleu4, nist2, nist4))

        # print samples
        ids = random.sample(list(range(len(all_pred))), result_num)
        print_samples(all_ref, all_pred, ids, self.tokenizer)

        return bleu2, bleu4, nist2, nist4


    def chatting(self, query):
        query = [self.tokenizer.sos_token_id] + self.tokenizer.encode(query)[:self.max_len-2] + [self.tokenizer.eos_token_id]
        query = query + [self.tokenizer.pad_token_id] * (self.max_len - len(query))
        
        query = torch.LongTensor(query).unsqueeze(0).to(self.device)
        trg = torch.LongTensor([self.tokenizer.sos_token_id]).unsqueeze(0).to(self.device)

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