"""
Transformer: the whole model (paper Figure 1, section 3.4)

Encoder reads, decoder writes, the Linear picks the word.
1. src_mask = pad_mask(src)                         built ONCE here, handed to the decoder
2. memory = encoder(src)                            [B, S_src, 128]
3. x = decoder(tgt_in, memory, src_mask)            [B, S_tgt, 128]
4. logits = out(x): Linear 128 -> 33 word scores    [B, S_tgt, 33]

No softmax here: CrossEntropyLoss (Stage 8) does it inside, argmax (Stage 9) doesn't need it.

Weight sharing (3.4): the paper shares ONE matrix between both embeddings and the Linear.
Ours can't be 3-way (13 digit rows vs 33 word rows), but the target embedding table and
the Linear weight are both [33, 128], so tie_weights=True makes them the SAME tensor.

Flashcard 72: src [1, 2], tgt_in [1, 3] -> logits [1, 3, 33]
"""

import torch
from torch import nn

from attention import pad_mask
from decoder import Decoder
from embeddings import D_model
from encoder import D_ff, N_layers, Encoder
from multi_head import N_heads


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=D_model, n_heads=N_heads, d_ff=D_ff,
                 n_layers=N_layers, dropout=0.1, pad_id=0, tie_weights=True):
        super().__init__()
        self.pad_id = pad_id
        # Stage 5 and Stage 6 bricks, unchanged, built by keyword
        self.encoder = Encoder(vocab_size=src_vocab_size, d_model=d_model, n_heads=n_heads, d_ff=d_ff,
                               n_layers=n_layers, dropout=dropout, pad_id=pad_id)      # 13 rows
        self.decoder = Decoder(vocab_size=tgt_vocab_size, d_model=d_model, n_heads=n_heads, d_ff=d_ff,
                               n_layers=n_layers, dropout=dropout, pad_id=pad_id)      # 33 rows
        # pre-softmax Linear: 128 numbers per word slot -> 33 word scores (weight [33, 128], bias [33])
        self.out = nn.Linear(d_model, tgt_vocab_size)

        if tie_weights:
            # 2-way sharing: target embedding table and Linear weight become ONE tensor
            table = self.decoder.embed.tok.table.weight                                # [33, 128]
            # smaller start (std 1/sqrt(128)) so the first scores are around 1, not around 11
            nn.init.normal_(table, mean=0.0, std=d_model ** -0.5)
            with torch.no_grad():
                table[pad_id].zero_()                                                  # <pad> row starts at 0 again
            self.out.weight = table                                                    # same object, not a copy

    def forward(self, src, tgt_in):
        # src: digit IDs [B, S_src]   tgt_in: word IDs, target shifted right [B, S_tgt]
        src_mask = pad_mask(src, self.pad_id)                                          # [B, 1, S_src], built once
        memory = self.encoder(src)                                                     # [B, S_src, 128]
        x = self.decoder(tgt_in, memory=memory, src_mask=src_mask)                     # [B, S_tgt, 128]
        return self.out(x)                                                             # logits [B, S_tgt, 33]