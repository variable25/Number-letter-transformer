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


if __name__ == "__main__":
    from data import PAD_ID, SRC_VOCAB, TGT_VOCAB, get_loaders

    def r(t):
        return [round(v, 3) for v in t.tolist()]

    torch.manual_seed(0)
    src_embed = Embedder(len(SRC_VOCAB), pad_id=PAD_ID).eval()   # eval: dropout off
    tgt_embed = Embedder(len(TGT_VOCAB), pad_id=PAD_ID).eval()

    # ---------- Sizes ----------
    print("--- Sizes ---")
    for name, m in [("src", src_embed), ("tgt", tgt_embed)]:
        n_weights = sum(p.numel() for p in m.parameters())
        print(f"{name} table {tuple(m.tok.table.weight.shape)} -> {n_weights} learned weights")
    print(f"barcode {tuple(src_embed.pos.pe.shape)} -> fixed, 0 learned weights")

    # ---------- A real training batch ----------
    src, tgt = next(iter(get_loaders()[0]))
    print("\n--- A real training batch: [B, S] IDs -> [B, S, 128] vectors ---")
    print(f"src {tuple(src.shape)} -> {tuple(src_embed(src).shape)}")
    print(f"tgt {tuple(tgt.shape)} -> {tuple(tgt_embed(tgt).shape)}")

    # ---------- Example: input 44 ----------
    # Both 4s fetch the same row from the table, so any difference in the output
    # can only come from the positional barcode.
    input_44 = torch.tensor([[7, 7]])            # ID 7 = digit "4"
    out = src_embed(input_44)[0]
    row_7 = src_embed.tok.table.weight[7, :4]
    scaled = row_7 * src_embed.tok.scale
    pe = src_embed.pos.pe

    print("\n--- Example: input 44 (first 4 of the 128 numbers shown) ---")
    print(f"Step 1  table row for '4' (ID 7):            {r(row_7)}")
    print(f"Step 2  x sqrt(128) = 11.314:                {r(scaled)}   <- same for both 4s")
    print(f"Step 3a barcode for position 0:              {r(pe[0, :4])}")
    print(f"Step 3b barcode for position 1:              {r(pe[1, :4])}")
    print(f"Result  first '4'  = step 2 + barcode pos 0: {r(out[0, :4])}")
    print(f"Result  second '4' = step 2 + barcode pos 1: {r(out[1, :4])}")
    print("Same digit, different position -> different vectors. The barcode is the only difference.")