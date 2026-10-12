""" 
Preparing the data for the model.
Process: 
1. Pick random numbers between 0-999,999
2. Split them if they are not single digit, assign IDs to each digit between 3-12, IDs 0, 1 and 2 are reserved for <pad>, <sos>, <eos>
3. Pick 128 batches of numbers(their IDs) and pad(their IDs) to match the highest digit numbers
This takes place in the CPU, not loaded into the GPU
Data is being prepared to be loaded into the GPU, that's why 128 numbers are picked and padded that way.

"""

import random

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

# ---------- 1. Ground truth: number -> words ----------
ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
        "seventeen", "eighteen", "nineteen"]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
MAX_N = 999_999

def _below_1000(n):
    words = []
    if n >= 100:
        words += [ONES[n // 100], "hundred"]
        n %= 100
    if n >= 20:
        words.append(TENS[n // 10])
        if n % 10:
            words.append(ONES[n % 10])
    elif n > 0:
        words.append(ONES[n])
    return words

def number_to_words(n):
    if n == 0:
        return "zero"
    words = []
    if n >= 1000:
        words += _below_1000(n // 1000) + ["thousand"]
        n %= 1000
    words += _below_1000(n)
    return " ".join(words)

# ---------- 2. Tokenizer / vocabulary: Writing numbers in literal words ----------
PAD, SOS, EOS = "<pad>", "<sos>", "<eos>"
SPECIALS = [PAD, SOS, EOS]
PAD_ID, SOS_ID, EOS_ID = 0, 1, 2

class Vocab:
    def __init__(self, tokens):
        self.itos = SPECIALS + sorted(set(tokens))          # id -> token
        self.stoi = {t: i for i, t in enumerate(self.itos)}  # token -> id

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens):
        return [self.stoi[t] for t in tokens]

    def decode(self, ids):
        out = []
        for i in ids:
            tok = self.itos[i]
            if tok == EOS:
                break
            if tok not in (PAD, SOS):
                out.append(tok)
        return out

SRC_VOCAB = Vocab(list("0123456789"))
TGT_VOCAB = Vocab(ONES + TENS[2:] + ["hundred", "thousand"])

def src_tokens(n):
    return list(str(n))

def tgt_tokens(n):
    return number_to_words(n).split()

# ---------- 3. Splits, Dataset, batching ----------
def make_splits(n_train=50_000, n_val=2_000, n_test=2_000, seed=42):
    rng = random.Random(seed)
    small = list(range(1000))  # 0-999: building blocks, all go to train
    n_big_train = n_train - len(small)
    big = rng.sample(range(1000, MAX_N + 1), n_big_train + n_val + n_test)
    train = small + big[:n_big_train]
    val = big[n_big_train:n_big_train + n_val]
    test = big[n_big_train + n_val:]
    return train, val, test

class NumberDataset(Dataset):
    def __init__(self, numbers):
        self.numbers = numbers

    def __len__(self):
        return len(self.numbers)

    def __getitem__(self, i):
        n = self.numbers[i]
        src = SRC_VOCAB.encode(src_tokens(n))
        tgt = [SOS_ID] + TGT_VOCAB.encode(tgt_tokens(n)) + [EOS_ID]
        return torch.tensor(src), torch.tensor(tgt)

def collate(batch):
    srcs, tgts = zip(*batch)
    src = pad_sequence(srcs, batch_first=True, padding_value=PAD_ID)
    tgt = pad_sequence(tgts, batch_first=True, padding_value=PAD_ID)
    return src, tgt

def get_loaders(batch_size=128):
    train, val, test = make_splits()
    return (
        DataLoader(NumberDataset(train), batch_size, shuffle=True, collate_fn=collate),
        DataLoader(NumberDataset(val), batch_size, collate_fn=collate),
        DataLoader(NumberDataset(test), batch_size, collate_fn=collate),
    )

