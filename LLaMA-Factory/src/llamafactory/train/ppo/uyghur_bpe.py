import re
from tokenizers import Tokenizer
from tokenizers.models import BPE
from typing import List, NamedTuple
from collections import defaultdict
import torch
from string import ascii_lowercase
from typing import List, Union
from tokenizers.pre_tokenizers import Whitespace
import numpy as np
from torch import Tensor


class Uyghur():
    def __init__(self, ):
        # self.uyghur_latin = "abcdefghijklmnopqrstuvwxyz éöü’" 
        # self._vocab_list = [self.pad_char, self.sos_char,self.eos_char] + list(self.uyghur_latin) # $ for padding char. index must be 0
        # self._vocab2idx = {v: idx for idx, v in enumerate(self._vocab_list)}
        tokenizer = Tokenizer(BPE())
        tokenizer.pre_tokenizer = Whitespace()
        self.tokenizer = tokenizer.from_file("/home/wyz/projects/LLaMA-Factory/src/llamafactory/train/ppo/tokenizer-trained.json")
        self.tokenizer.add_tokens([' '])
        self.ind2char = {v: k for k, v in self.tokenizer.get_vocab().items()}

    def __len__(self):
        return len(self.ind2char)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text

    def encode(self, text):
        return self.tokenizer.encode(text).ids


    # def encode(self, s):
    #     s = s.replace("-", ' ').replace(",", ' ').replace(".", ' ').replace("!", ' ').replace("?", ' ').replace("'","’")
    #     s = re.sub('\s+',' ',s).strip().lower()
    #     seq = [self.vocab_to_idx(v) for v in s if v in self.uyghur_latin]
    #     return seq


    def decode(self, seq):
        # print("seq_bpe")
        # print(seq)
        vocabs = []
        for idx in seq:
            # print("*"*20)
            # print(idx.item())
            v = self.idx_to_vocab(idx)
            if idx == self.pad_idx or  idx == self.eos_idx:
                break
            elif idx == self.sos_idx:
                pass
            else:
                vocabs.append(v)
        s = re.sub('\s+',' ',"".join(vocabs)).strip()
        return s

    # def vocab_to_idx(self, vocab):
    #     return self._vocab2idx[vocab]

    def idx_to_vocab(self, idx):
        return self.ind2char[idx]

    # def vocab_list(self):
    #     return self._vocab_list

    @property
    def vocab_size(self):
        return len(self.ind2char)

    @property
    def pad_idx(self):
        return 0

    @property
    def sos_idx(self):
        return 1

    @property
    def eos_idx(self):
        return 2

    @property
    def pad_char(self):
        return "<pad>"

    @property
    def sos_char(self):
        return "<sos>"

    @property
    def eos_char(self):
        return "<eos>"

uyghur_bpe = Uyghur()
