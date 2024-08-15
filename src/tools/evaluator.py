import numpy as np
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from nltk.translate.nist_score import corpus_nist, sentence_nist



class Evaluator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer


    def cal_bleu_score(self, pred, gt, n=4):
        assert (isinstance(pred, str) and isinstance(gt, str)) or (isinstance(pred, list) and isinstance(gt, list))    
        if isinstance(pred, str):
            pred, gt = [pred], [gt]
        weights = tuple([1/n] * n)
        pred = [self.tokenizer.tokenize(text) for text in pred]
        gt = [[self.tokenizer.tokenize(text)] for text in gt]
        if isinstance(pred, str):
            return sentence_bleu(gt[0], pred[0], weights=weights)
        return corpus_bleu(gt, pred, weights=weights)
    
    
    def cal_nist_score(self, pred, gt, n=4):
        assert (isinstance(pred, str) and isinstance(gt, str)) or (isinstance(pred, list) and isinstance(gt, list))    
        if isinstance(pred, str):
            pred, gt = [pred], [gt]
        pred = [self.tokenizer.tokenize(text) for text in pred]
        gt = [[self.tokenizer.tokenize(text)] for text in gt]
        if isinstance(pred, str):
            return sentence_nist(gt[0], pred[0], n=n)
        return corpus_nist(gt, pred, n=n)
    

    def cal_ppl(self, loss):
        return np.exp(loss)
    

    