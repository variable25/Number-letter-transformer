import math

import pytest
import torch

from data import PAD_ID, SRC_VOCAB, TGT_VOCAB, NumberDataset, collate
from embeddings import D_model, Embedder, PositionalEncoding, TokenEmbedding, sin_table


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


# ---------- Token embedding (section 3.4) ----------
def test_token_embedding_shape():
    emb = TokenEmbedding(len(SRC_VOCAB))
    ids = torch.tensor([[10, 5, 0, 0], [7, 3, 10, 5]])   # 72 (padded), 4072
    assert emb(ids).shape == (2, 4, D_model)


def test_lookup_is_row_times_sqrt_d_model():
    emb = TokenEmbedding(len(SRC_VOCAB))
    out = emb(torch.tensor([[7]]))
    assert torch.allclose(out[0, 0], emb.table.weight[7] * math.sqrt(D_model))


def test_same_id_gives_same_vector():
    emb = TokenEmbedding(len(SRC_VOCAB))
    out = emb(torch.tensor([[7, 3, 7]]))                 # "4" appears twice
    assert torch.equal(out[0, 0], out[0, 2])


def test_pad_row_is_zero_and_never_trained():
    emb = TokenEmbedding(len(SRC_VOCAB), pad_id=PAD_ID)
    assert torch.count_nonzero(emb.table.weight[PAD_ID]) == 0
    emb(torch.tensor([[7, PAD_ID]])).sum().backward()
    assert torch.count_nonzero(emb.table.weight.grad[PAD_ID]) == 0   # pad row gets no nudge
    assert torch.count_nonzero(emb.table.weight.grad[7]) > 0         # real token does


# ---------- Positional barcode (section 3.5) ----------
def test_barcode_matches_paper_formula():
    pe = sin_table(max_len=20, d_model=D_model)
    for pos in (0, 1, 5, 19):
        for i in (0, 1, 10, 63):                         # pair i covers dims 2i (sin) and 2i+1 (cos)
            angle = pos / 10000 ** (2 * i / D_model)
            assert pe[pos, 2 * i].item() == pytest.approx(math.sin(angle), abs=1e-5)
            assert pe[pos, 2 * i + 1].item() == pytest.approx(math.cos(angle), abs=1e-5)


def test_barcode_bounded_and_every_position_unique():
    pe = sin_table()
    assert pe.abs().max() <= 1.0
    dist = torch.cdist(pe, pe).fill_diagonal_(float("inf"))
    assert dist.min() > 0.1                              # no two positions share a barcode


def test_barcode_is_fixed_not_learned():
    pos = PositionalEncoding()
    assert list(pos.parameters()) == []
    assert "pe" in pos.state_dict()                      # still saved with the model


def test_eval_mode_adds_barcode_exactly():
    pos = PositionalEncoding().eval()
    x = torch.randn(2, 5, D_model)
    assert torch.allclose(pos(x), x + pos.pe[:5])


# ---------- Dropout (section 5.4) ----------
def test_dropout_only_in_training():
    pos = PositionalEncoding(dropout=0.1)
    x = torch.full((64, 11, D_model), 5.0)               # x + pe >= 4, so any zero came from dropout
    pos.train()
    zero_frac = (pos(x) == 0).float().mean().item()
    assert 0.08 < zero_frac < 0.12
    pos.eval()
    assert (pos(x) == 0).sum() == 0
    assert torch.equal(pos(x), pos(x))


# ---------- Full embedder ----------
def test_4072_and_2704_differ_only_because_of_position():
    emb = Embedder(len(SRC_VOCAB)).eval()
    a = torch.tensor([[7, 3, 10, 5]])                    # 4072
    b = torch.tensor([[5, 10, 3, 7]])                    # 2704 = 4072 reversed
    assert torch.allclose(emb.tok(b).flip(1), emb.tok(a))   # same digits: identical vectors, reordered
    assert not torch.allclose(emb(b).flip(1), emb(a))       # barcode makes the order visible


def test_real_batch_shapes_and_weight_counts():
    src, tgt = collate([NumberDataset([n])[0] for n in (72, 4072, 999_999)])
    src_embed = Embedder(len(SRC_VOCAB), pad_id=PAD_ID)
    tgt_embed = Embedder(len(TGT_VOCAB), pad_id=PAD_ID)
    assert src_embed(src).shape == (3, 6, D_model)       # 999999 -> 6 digits
    assert tgt_embed(tgt).shape == (3, 11, D_model)      # <sos> + 9 words + <eos>
    assert sum(p.numel() for p in src_embed.parameters()) == 13 * D_model
    assert sum(p.numel() for p in tgt_embed.parameters()) == 33 * D_model


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
def test_runs_on_gpu():
    emb = Embedder(len(SRC_VOCAB)).to("cuda")
    assert emb.pos.pe.device.type == "cuda"              # buffer moved along with the weights
    assert emb(torch.tensor([[7, 3, 10, 5]], device="cuda")).device.type == "cuda"