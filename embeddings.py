"""
Embedding the tokens representing the data

The tokens or IDs representing the input text are now assigned a list of 128 numbers
The Steps performed in this phase are:
1. Look up each ID's learned row or weights/list of numbers in the embedding table
2. Scale by sqrt(d_model=128) so very high or low magnitude do not affect the average
3. Add position embeddings as well
4. For training, define the dropout

"""

import math

import torch
from torch import nn

D_model = 128
Max_len = 64

class TokenEmbedding(nn.Module):
    def __init__(self,vocab_size,d_model=D_model,pad_id=0):
        super().__init__()
        self.table = nn.Embedding(vocab_size,d_model,padding_idx=pad_id)
        self.scale = math.sqrt(d_model)

    def forward(self, ids):
        return self.table(ids) * self.scale

def sin_table(max_len=Max_len, d_model=D_model):
    # PE(pos, 2i) = sin(pos/10000^(2i/d_model)) -> for even column nos, PE(pos,2i+1) = cos(pos/10000^(2i/d_model)) for odd col nums
    # pos -> row number, i is column no
    pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
    two_i = torch.arange(0,d_model,2,dtype=torch.float)
    angle = pos / (10000 ** (two_i/d_model))
    pe = torch.zeros(max_len, d_model) # tensors with zeros
    pe[:,0::2] = torch.sin(angle)
    pe[:,1::2] = torch.cos(angle)
    return pe

class PositionalEncoding(nn.Module):
    # Performs the positional encoding i.e a list embedded to denote the unique positions of each token
    def __init__(self, d_model=D_model, max_len=Max_len, dropout=0.1):
        super().__init__()
        # buffer: saved with the model and moved to the GPU with it, but NOT a weight (never nudged)
        self.register_buffer("pe", sin_table(max_len, d_model))
        # to stop the model from memorizing, we change some values to 0
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):                        # [B, S, d_model]
        S = x.size(1)
        return self.dropout(x + self.pe[:S])     # pe[:S] is [S, d_model], same barcode for every card


class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model=D_model, max_len=Max_len, dropout=0.1, pad_id=0):
        super().__init__()
        self.tok = TokenEmbedding(vocab_size, d_model, pad_id)
        self.pos = PositionalEncoding(d_model, max_len, dropout)

    def forward(self, ids):                      # [B, S]
        return self.pos(self.tok(ids))           # [B, S, d_model]


