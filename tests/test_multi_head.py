import pytest
import torch
from torch import nn

from attention import attention, causal_mask, pad_mask
from data import SOS_ID, SRC_VOCAB, TGT_VOCAB, NumberDataset, collate, src_tokens, tgt_tokens
from embeddings import Embedder
from multi_head import MultiHeadAttention


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


def _src(*numbers):
    return collate([NumberDataset([n])[0] for n in numbers])[0]


def _x_4072():
    # real pipeline: 4072 -> IDs -> Embedder -> [1, 4, 128]
    ids = torch.tensor([SRC_VOCAB.encode(src_tokens(4072))])
    return Embedder(len(SRC_VOCAB)).eval()(ids)


# ---------- Sizes ----------
def test_sizes():
    mha = MultiHeadAttention()
    assert mha.n_heads == 8 and mha.d_k == 16
    for table in (mha.W_Q, mha.W_K, mha.W_V, mha.W_O):
        assert table.weight.shape == (128, 128)
        assert table.bias is None                            # paper: no bias
    assert sum(p.numel() for p in mha.parameters()) == 4 * 128 * 128


# ---------- Split and join (.view / .transpose) ----------
def test_each_head_gets_its_own_16_numbers():
    x = _x_4072()
    heads = MultiHeadAttention().split_heads(x)
    assert heads.shape == (1, 8, 4, 16)
    assert torch.equal(heads[0, 0], x[0, :, 0:16])           # head 0: numbers 0-15 of every digit
    assert torch.equal(heads[0, 1], x[0, :, 16:32])          # head 1: numbers 16-31
    assert torch.equal(heads[0, 7], x[0, :, 112:128])        # head 7: numbers 112-127


def test_join_undoes_split():
    x = _x_4072()
    mha = MultiHeadAttention()
    assert torch.equal(mha.join_heads(mha.split_heads(x)), x)


# ---------- The formula (section 3.2.2) ----------
def test_4072_shapes_and_weights():
    x = _x_4072()
    mha = MultiHeadAttention()
    out = mha(x, x, x)
    assert out.shape == (1, 4, 128)                          # same shape in and out
    assert mha.weights.shape == (1, 8, 4, 4)                 # 8 heads, 4 searching rows x 4 titles
    assert torch.allclose(mha.weights.sum(-1), torch.ones(1, 8, 4))


def test_4072_matches_formula_head_by_head():
    # the paper's formula written out slowly: one head at a time with Stage 3 attention()
    x = _x_4072()
    mha = MultiHeadAttention()
    q = x @ mha.W_Q.weight.T                                 # nn.Linear stores its table flipped, .T flips it back
    k = x @ mha.W_K.weight.T
    v = x @ mha.W_V.weight.T
    heads = []
    for h in range(8):
        cols = slice(16 * h, 16 * (h + 1))                   # head h owns numbers 16h .. 16h+15
        head_out, _ = attention(q[..., cols], k[..., cols], v[..., cols])
        heads.append(head_out)
    expected = torch.cat(heads, dim=-1) @ mha.W_O.weight.T   # Concat(head_0..head_7) W_O
    assert torch.allclose(mha(x, x, x), expected, atol=1e-4)


def test_matches_pytorch_builtin():
    # PyTorch's own version as a referee, given our exact 4 tables
    x = _x_4072()
    ours = MultiHeadAttention()
    theirs = nn.MultiheadAttention(128, 8, bias=False, batch_first=True)
    with torch.no_grad():
        theirs.in_proj_weight.copy_(torch.cat([ours.W_Q.weight, ours.W_K.weight, ours.W_V.weight]))
        theirs.out_proj.weight.copy_(ours.W_O.weight)
    their_out, their_w = theirs(x, x, x, average_attn_weights=False)
    assert torch.allclose(ours(x, x, x), their_out, atol=1e-4)
    assert torch.allclose(ours.weights, their_w, atol=1e-4)


# ---------- Masks inside every head ----------
def test_pad_columns_get_zero_in_every_head():
    src = _src(72, 4072)                                     # 72 -> "7" "2" <pad> <pad>
    x = Embedder(len(SRC_VOCAB)).eval()(src)
    mha = MultiHeadAttention()
    mha(x, x, x, pad_mask(src))
    assert (mha.weights[0, :, :, 2:] == 0).all()             # all 8 heads ignore 72's pads


def test_pad_videos_cannot_change_output():
    src = _src(72, 4072)
    x = Embedder(len(SRC_VOCAB)).eval()(src)
    mha = MultiHeadAttention()
    out1 = mha(x, x, x, pad_mask(src))
    junk = x.clone()
    junk[0, 2:] = 999.0                                      # scribble on 72's two pad positions
    out2 = mha(x, junk, junk, pad_mask(src))                 # titles and videos made from the scribbles
    assert torch.allclose(out1[0, :2], out2[0, :2], atol=1e-4)   # rows for "7" and "2" unaffected


def test_future_words_cannot_change_past_in_any_head():
    # decoder input for 72: "<sos> seventy two" (the words shifted right, from Stage 1)
    ids = torch.tensor([[SOS_ID] + TGT_VOCAB.encode(tgt_tokens(72))])   # 3 tokens
    y = Embedder(len(TGT_VOCAB)).eval()(ids)                 # [1, 3, 128]
    mha = MultiHeadAttention()
    out1 = mha(y, y, y, causal_mask(3))
    assert (mha.weights[0].triu(1) == 0).all()               # above the diagonal covered in all 8 heads
    later = y.clone()
    later[0, 2] = 999.0                                      # rewrite "two" (the answer)
    out2 = mha(later, later, later, causal_mask(3))
    assert torch.allclose(out1[0, :2], out2[0, :2], atol=1e-4)   # "<sos>" and "seventy" never saw it


def test_decoder_can_search_encoder():
    # Stage 6 preview: target words search the source digits (q_in differs from k_in, v_in)
    src = torch.tensor([SRC_VOCAB.encode(src_tokens(72))])                 # "7" "2" -> 2 tokens
    tgt = torch.tensor([[SOS_ID] + TGT_VOCAB.encode(tgt_tokens(72))])      # "<sos> seventy two" -> 3 tokens
    digits = Embedder(len(SRC_VOCAB)).eval()(src)            # [1, 2, 128]
    words = Embedder(len(TGT_VOCAB)).eval()(tgt)             # [1, 3, 128]
    mha = MultiHeadAttention()
    out = mha(words, digits, digits)
    assert out.shape == (1, 3, 128)                          # one answer per word
    assert mha.weights.shape == (1, 8, 3, 2)                 # 3 words searching 2 digits, in 8 heads


def test_all_four_tables_can_be_nudged():
    # forward -> loss -> backward: every table must receive a gradient
    x = _x_4072()
    mha = MultiHeadAttention()
    mha(x, x, x).pow(2).mean().backward()
    for table in (mha.W_Q, mha.W_K, mha.W_V, mha.W_O):
        assert table.weight.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
def test_runs_on_gpu():
    x = _x_4072().to("cuda")
    mha = MultiHeadAttention().to("cuda")
    out = mha(x, x, x, causal_mask(4, device="cuda"))
    assert out.device.type == "cuda"