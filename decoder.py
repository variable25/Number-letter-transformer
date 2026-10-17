"""
Decoder (paper section 3.1 decoder stack, section 3.2.3 uses of attention)

Encoder reads, decoder writes.
One decoder layer = 3 sublayers, each followed by Add & Norm:
1. Masked self-attention: words look at earlier words only   (tgt_mask = pad + causal)
2. Cross-attention: words search the digits in memory         (src_mask = pad of digits)
   Q from the words, K and V from the encoder's memory
3. Feed-forward: each word thinks alone                       (same FeedForward as Stage 5)

x = norm1(x + dropout(self_attn(x, x, x, tgt_mask)))
x = norm2(x + dropout(cross_attn(x, memory, memory, src_mask)))
x = norm3(x + dropout(ffn(x)))

Flashcard 72: tgt_in [B, 3] -> output [B, 3, 128], memory [B, 2, 128]
Picking the actual word (Linear 128 -> 33) is Stage 7.
"""

from torch import nn

from attention import causal_mask, pad_mask
from embeddings import D_model, Embedder
from encoder import D_ff, N_layers, FeedForward
from multi_head import MultiHeadAttention, N_heads

# One decoder layer: 3 sublayers, 3 Add & Norms
class DecoderLayer(nn.Module):
    def __init__(self, d_model=D_model, n_heads=N_heads, d_ff=D_ff, dropout=0.1):
        super().__init__()
        # Two seperate attention bricks , each with it's own W_Q, W_K, W_V, W_O
        self.self_attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads) # Words look at words
        self.cross_attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads) # Words search digits
        self.ffn = FeedForward(d_model=d_model, d_ff=d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        # x: words [B, S_tgt, 128]   memory: digits from the encoder [B, S_src, 128]
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))                   # look back
        x = self.norm2(x + self.dropout(self.cross_attn(x, memory, memory, src_mask)))        # search digits
        x = self.norm3(x + self.dropout(self.ffn(x)))                                         # think alone
        return x                                                                              # [B, S_tgt, 128]


# The whole writing process takes place in this class
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model=D_model, n_heads=N_heads, d_ff=D_ff,
                 n_layers=N_layers, dropout=0.1, pad_id=0):
        super().__init__()
        self.pad_id = pad_id
        self.embed = Embedder(vocab_size=vocab_size, d_model=d_model, dropout=dropout, pad_id=pad_id)   # 33 rows
        # 3 SEPARATE layers, each with its own random weights, built by keyword
        self.layers = nn.ModuleList([
            DecoderLayer(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(n_layers)
        ])

    def forward(self, tgt_in, memory, src_mask):
        # tgt_in: word IDs [B, S_tgt] (target shifted right)   memory: [B, S_src, 128]
        # src_mask: pad_mask(src) [B, 1, S_src], built by the caller (Encoder only returns memory)
        S_tgt = tgt_in.size(1)
        # covered if the word is <pad> OR in the future: [B, 1, S_tgt] & [S_tgt, S_tgt] -> [B, S_tgt, S_tgt]
        tgt_mask = pad_mask(tgt_in, self.pad_id) & causal_mask(S_tgt, device=tgt_in.device)
        x = self.embed(tgt_in)                                   # [B, S_tgt, 128]
        for layer in self.layers:                                # all 3 layers search the SAME memory
            x = layer(x, memory=memory, src_mask=src_mask, tgt_mask=tgt_mask)
        return x