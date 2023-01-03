import torch
import os
import pickle
import sys
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.nist_score import corpus_nist



"""
common utils
"""
def load_dataset(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def check_data(base_path):
    processed_path = {
        'train': base_path+'data/dailydialog/processed/dailydialog.train',
        'val': base_path+'data/dailydialog/processed/dailydialog.val',
        'test': base_path+'data/dailydialog/processed/dailydialog.test'
        }

    if not (os.path.isfile(processed_path['train']) and os.path.isfile(processed_path['val']) and os.path.isfile(processed_path['test'])):
        print('Please prepare the dataset..')
        sys.exit()
      

def make_dataset_path(base_path):
    processed_path = {
        'train': base_path+'data/dailydialog/processed/dailydialog.train',
        'val': base_path+'data/dailydialog/processed/dailydialog.val',
        'test': base_path+'data/dailydialog/processed/dailydialog.test'
        }
    return processed_path


def save_checkpoint(file, model, optimizer):
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(state, file)
    print('model pt file is being saved\n')



def bleu_score(ref, pred, weights):
    smoothing = SmoothingFunction().method3
    return corpus_bleu(ref, pred, weights, smoothing)


def nist_score(ref, pred, n):
    return corpus_nist(ref, pred, n)


def cal_scores(ref, pred, type, n_gram):
    assert type in ['bleu', 'nist']
    if type == 'bleu':
        wts = tuple([1/n_gram]*n_gram)
        return bleu_score(ref, pred, wts)
    return nist_score(ref, pred, n_gram)


def tensor2list(ref, pred, tokenizer):
    ref, pred = torch.cat(ref, dim=0)[:, 1:], torch.cat(pred, dim=0)[:, :-1]
    ref = [[tokenizer.tokenize(tokenizer.decode(ref[i].tolist()))] for i in range(ref.size(0))]
    pred = [tokenizer.tokenize(tokenizer.decode(pred[i].tolist())) for i in range(pred.size(0))]
    return ref, pred


def print_samples(ref, pred, ids, tokenizer):
    print('-'*50)
    for i in ids:
        r, p = tokenizer.tokenizer.convert_tokens_to_string(ref[i][0]), tokenizer.tokenizer.convert_tokens_to_string(pred[i])
        print('gt  : {}'.format(r))
        print('pred: {}\n'.format(p))
    print('-'*50 + '\n')


def preprocessing_query(queries, tokenizer):
    total_s = [tokenizer.cls_token_id]
    for s in queries:
        total_s += tokenizer.encode(s) + [tokenizer.sep_token_id]
    return total_s