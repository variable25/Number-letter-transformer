"""
Scaled dot-product attention (paper section 3.2.1, Figure 2 left)

Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

Q (search): what each token is looking for
K (titles): what each token advertises
V (videos): what each token hands over

Steps:
1. Score every search against every title: Q @ K^T -> [B, S_q, S_k]
2. Divide by sqrt(d_k) so big scores don't make softmax all-or-nothing
3. Mask (optional): covered cells get -inf, softmax turns them into 0
4. Softmax along each row -> weights, each row sums to 1
5. Weights @ V -> each token's output is a blend of the videos

No learned weights here. W_Q, W_K, W_V come in Stage 4 (multi-head).
"""

import math

import torch

from data import SRC_VOCAB, NumberDataset, collate, src_tokens
from embeddings import Embedder

def attention(q, k, v, mask=None):
    # q: [B, S_q, d_k]   k, v: [B, S_k, d_k]   (extra dims like heads in front are fine)
    # mask: True = allowed to look, False = covered. Must broadcast to [B, S_q, S_k]
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)       # [B, S_q, S_k]
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))   # covered answers
    weights = scores.softmax(dim=-1)                         # along each row, sums to 1
    return weights @ v, weights                              # [B, S_q, d_k], [B, S_q, S_k]


def pad_mask(ids, pad_id=0):
    # [B, S] -> [B, 1, S]: column j is covered if token j is <pad>
    # the 1 means the same mask row is reused by every searching row
    return (ids != pad_id).unsqueeze(1)


def causal_mask(S, device=None):
    # [S, S] lower triangle: row i may look at columns 0..i only
    # the diagonal is always True, so no row is ever fully covered (that would give NaN)
    return torch.tril(torch.ones(S, S, dtype=torch.bool, device=device))


