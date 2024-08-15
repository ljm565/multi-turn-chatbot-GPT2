from transformers import GPT2Tokenizer



class CustomGPT2Tokenizer:
    def __init__(self, config):
        self.pretrained_model = config.pretrained_model
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.pretrained_model)
        self.tokenizer.add_special_tokens({'cls_token': '[CLS]', 'sep_token': '[SEP]', 'pad_token': '[PAD]'})
        
        self.pad_token, self.pad_token_id = self.tokenizer.pad_token, self.tokenizer.pad_token_id
        self.cls_token, self.cls_token_id = self.tokenizer.cls_token, self.tokenizer.cls_token_id
        self.sep_token, self.sep_token_id = self.tokenizer.sep_token, self.tokenizer.sep_token_id
        self.unk_token, self.unk_token_id = self.tokenizer.unk_token, self.tokenizer.unk_token_id

        self.vocab_size = len(self.tokenizer)


    def tokenize(self, s):
        return self.tokenizer.tokenize(s)


    def encode(self, s):
        return self.tokenizer.encode(s)


    def decode(self, tok):
        try:
            tok = tok[:(len(tok) - list(reversed(tok)).index(self.sep_token_id))]
        except ValueError:
            try:
                tok = tok[:tok.index(self.pad_token_id)]
            except:
                pass
        return self.tokenizer.decode(tok)