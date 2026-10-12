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


if __name__ == "__main__":
    from data import NumberDataset, collate

    def r(t):
        return [round(v, 3) for v in t.tolist()]

        # ---------- Example 1: the number 4072, real 128 numbers per digit ----------
    # Real pipeline from Stages 1-2: digits -> IDs -> Embedder (table row x sqrt(128) + barcode).
    # The model is untrained, so the 128 numbers are random starting weights.
    # Each digit uses its same 128 numbers as search (Q), title (K) and video (V);
    # Stage 4 will give each its own version.
    torch.manual_seed(0)
    digits = src_tokens(4072)                                   # ["4", "0", "7", "2"]
    ids = torch.tensor([SRC_VOCAB.encode(digits)])              # [[7, 3, 10, 5]]
    embedder = Embedder(len(SRC_VOCAB)).eval()                  # eval: dropout off
    x = embedder(ids)                                           # 4 digits x 128 numbers
    out, w = attention(x, x, x)
    dots = (x @ x.transpose(-2, -1))[0]                         # step 1 only, for printing

    print("--- Example 1: 4072, 128 numbers per digit (untrained) ---")
    print(f"digits {digits} -> IDs {ids[0].tolist()}")
    print(f"after Embedder: shape {list(x.shape)}  (1 number, 4 digits, 128 numbers each)")
    print(f"'7' before attention, first 4 numbers out of 128 embedded to it: {r(x[0, 2, :4])}")
    print(f"\nStep 1  dot products, '7' looking at these numbers: 4 / 0 / 7 / 2: {r(dots[2])}")
    print(f"Step 2  divided by sqrt(128) = 11.31:              {r(dots[2] / math.sqrt(128))}")
    print("Step 3  mask: none (4072 has no pads)")
    print("Step 4  softmax -> percentages, each digit looking at 4 / 0 / 7 / 2:")
    for i, d in enumerate(digits):
        print(f"        '{d}': {r(w[0, i])}")
    print(f"Step 5  output shape {list(out.shape)}  <- same shape as the input")
    print(f"'7' after attention, first 4 of 128:  {r(out[0, 2, :4])}")
    # ---------- Example 2: causal mask on the target "<sos> seventy two" ----------
    # All Q and K are zeros, so every score is equal (0). The ONLY thing shaping
    # the weights is the mask, which makes its effect easy to see.
    zeros = torch.zeros(1, 3, 2)
    _, w = attention(zeros, zeros, zeros, causal_mask(3))
    print("\n--- Example 2: causal mask, target '<sos> seventy two', all scores equal ---")
    print("row = the word doing the search, column = the word being looked at")
    for i, word in enumerate(["<sos>", "seventy", "two"]):
        print(f"row {i} {word:>8}: {r(w[0, i])}")

    # ---------- Example 3: pad mask, "72" batched with "4072" ----------
    # Real batching from data.py: 72 gets two <pad> so it matches 4072's 4 tokens.
    # All scores equal again, so only the pad mask shapes the weights.
    src, _ = collate([NumberDataset([n])[0] for n in (72, 4072)])
    zeros = torch.zeros(2, 4, 2)
    _, w = attention(zeros, zeros, zeros, pad_mask(src))
    print("\n--- Example 3: pad mask, '72' batched with '4072', all scores equal ---")
    print(f"src IDs for 72:    {src[0].tolist()}   <- two <pad> (ID 0)")
    print(f"pad_mask for 72:   {pad_mask(src)[0, 0].tolist()}")
    print(f"72,   row 0 ('7'): {r(w[0, 0])}   <- pad columns get 0")
    print(f"4072, row 0 ('4'): {r(w[1, 0])}   <- no pads, shared 4 ways")