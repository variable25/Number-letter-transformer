"""
Multi-head attention (paper section 3.2.2, Figure 2 right)

MultiHead(Q, K, V) = Concat(head_0, ..., head_7) W_O
    head_i = Attention(Q W_Q_i, K W_K_i, V W_V_i)

Stage 3 problem: Q = K = V = the same 128 numbers, so each digit mostly found itself.
Fix 1: three learned tables turn the same 128 numbers into a separate
       search (W_Q), title (W_K) and video (W_V).
Fix 2: cut the 128 numbers into 8 heads of 16, so 8 searches run side by side.

Steps:
1. Project: x @ W_Q, x @ W_K, x @ W_V        [B, S, 128] each
2. Split into 8 heads of 16 numbers          [B, 8, S, 16]
3. Stage 3 attention() inside every head      out [B, 8, S_q, 16], weights [B, 8, S_q, S_k]
4. Glue the 8 heads back together             [B, S_q, 128]
5. Mix the heads with W_O                     [B, S_q, 128]
"""

from torch import nn

from attention import attention
from embeddings import D_model

N_heads = 8


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=D_model, n_heads=N_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must split evenly into heads"
        self.n_heads = n_heads
        self.d_k = d_model // n_heads                        # 128 // 8 = 16 numbers per head
        # 4 learned tables, 128 x 128 each, random at the start
        # bias=False: the paper's formula has no bias, just the tables
        self.W_Q = nn.Linear(d_model, d_model, bias=False)   # makes the searches
        self.W_K = nn.Linear(d_model, d_model, bias=False)   # makes the titles
        self.W_V = nn.Linear(d_model, d_model, bias=False)   # makes the videos
        self.W_O = nn.Linear(d_model, d_model, bias=False)   # mixes the 8 heads' results
        self.weights = None                                  # last attention weights, saved for Stage 10 maps

    def split_heads(self, x):                                # [B, S, 128] -> [B, 8, S, 16]
        B, S, _ = x.shape
        x = x.view(B, S, self.n_heads, self.d_k)             # cut 128 into 8 x 16: [B, S, 8, 16]
        return x.transpose(1, 2)                             # heads in front:   [B, 8, S, 16]

    def join_heads(self, x):                                 # [B, 8, S, 16] -> [B, S, 128]
        B, _, S, _ = x.shape
        x = x.transpose(1, 2).contiguous()                   # back to [B, S, 8, 16], tidied in memory
        return x.view(B, S, self.n_heads * self.d_k)         # glue 8 x 16 into 128

    def forward(self, q_in, k_in, v_in, mask=None):
        # q_in:       [B, S_q, 128] the tokens doing the searching
        # k_in, v_in: [B, S_k, 128] the tokens being searched (always the same tensor for us)
        # self-attention: q_in = k_in = v_in. Decoder searching the encoder (Stage 6): q_in differs.
        if mask is not None and mask.dim() == 3:
            mask = mask.unsqueeze(1)                         # [B, 1, S_k] -> [B, 1, 1, S_k]: one mask for all heads
        q = self.split_heads(self.W_Q(q_in))                 # searches [B, 8, S_q, 16]
        k = self.split_heads(self.W_K(k_in))                 # titles   [B, 8, S_k, 16]
        v = self.split_heads(self.W_V(v_in))                 # videos   [B, 8, S_k, 16]
        out, weights = attention(q, k, v, mask)              # Stage 3, run in all 8 heads at once
        self.weights = weights.detach()                      # a copy to look at, not part of learning
        return self.W_O(self.join_heads(out))                # [B, S_q, 128]