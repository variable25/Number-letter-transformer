import math

import pytest
import torch
import torch.nn.functional as F

from attention import attention, causal_mask, pad_mask
from data import NumberDataset, collate

from data import SRC_VOCAB, src_tokens
from embeddings import Embedder


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


def _src(*numbers):
    return collate([NumberDataset([n])[0] for n in numbers])[0]


# ---------- The formula (section 3.2.1) ----------
def test_output_and_weight_shapes():
    # decoder searching the encoder: 11 target rows over 6 source columns
    q = torch.randn(2, 11, 16)
    k = torch.randn(2, 6, 16)
    v = torch.randn(2, 6, 16)
    out, w = attention(q, k, v)
    assert out.shape == (2, 11, 16)                      # one blended video per row
    assert w.shape == (2, 11, 6)                         # rows = searches, columns = titles


def test_every_row_of_weights_sums_to_one():
    x = torch.randn(4, 5, 16)
    _, w = attention(x, x, x)
    assert torch.allclose(w.sum(-1), torch.ones(4, 5))
    assert (w >= 0).all()


def test_4072_follows_the_formula_step_by_step():
    # real pipeline: 4072 -> IDs -> Embedder -> 4 digits x 128 numbers
    ids = torch.tensor([SRC_VOCAB.encode(src_tokens(4072))])
    x = Embedder(len(SRC_VOCAB)).eval()(ids)[0]                 # 4 digits x 128 numbers
    out, w = attention(x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0))
    seven = x[2]
    # step 1-2 by hand: the '7' dotted with each digit, divided by sqrt(128)
    scores_for_7 = torch.stack([(seven * x[j]).sum() for j in range(4)]) / math.sqrt(128)
    assert torch.allclose(w[0, 2], scores_for_7.softmax(-1), atol=1e-5)
    # step 5 by hand: new '7' = each digit's 128 numbers times its percentage, added up
    new_seven = sum(w[0, 2, j] * x[j] for j in range(4))
    assert torch.allclose(out[0, 2], new_seven, atol=1e-4)


def test_matches_pytorch_builtin():
    q, k, v = torch.randn(2, 5, 16), torch.randn(2, 5, 16), torch.randn(2, 5, 16)
    ours, _ = attention(q, k, v)
    assert torch.allclose(ours, F.scaled_dot_product_attention(q, k, v), atol=1e-5)
    ours, _ = attention(q, k, v, causal_mask(5))         # PyTorch also uses True = allowed
    theirs = F.scaled_dot_product_attention(q, k, v, attn_mask=causal_mask(5))
    assert torch.allclose(ours, theirs, atol=1e-5)


# ---------- Pad mask ----------
def test_pad_mask_shape_and_values():
    src = _src(72, 4072)                                 # 72 -> "7" "2" <pad> <pad>
    m = pad_mask(src)
    assert m.shape == (2, 1, 4)                          # one mask row, shared by every searching row
    assert m[0, 0].tolist() == [True, True, False, False]
    assert m[1, 0].all()


def test_pad_columns_get_zero_weight():
    src = _src(72, 4072)
    x = torch.randn(2, 4, 16)
    _, w = attention(x, x, x, pad_mask(src))
    assert (w[0, :, 2:] == 0).all()                      # nobody looks at 72's pads
    assert (w[1] > 0).all()                              # 4072 has no pads


def test_pad_keys_cannot_change_output():
    src = _src(72, 4072)
    x = torch.randn(2, 4, 16)
    out1, _ = attention(x, x, x, pad_mask(src))
    junk = x.clone()
    junk[0, 2:] = 999.0                                  # scribble on 72's pad columns
    out2, _ = attention(x, junk, junk, pad_mask(src))
    assert torch.allclose(out1[0, :2], out2[0, :2])      # rows for "7" and "2" unaffected


# ---------- Causal mask (section 3.2.3) ----------
def test_causal_mask_is_lower_triangle():
    assert causal_mask(3).tolist() == [
        [True, False, False],                            # row 0 sees column 0 only
        [True, True, False],
        [True, True, True],
    ]


def test_future_cannot_change_past():
    x = torch.randn(1, 4, 16)
    out1, w = attention(x, x, x, causal_mask(4))
    assert (w[0].triu(1) == 0).all()                     # everything above the diagonal covered
    later = x.clone()
    later[0, 3] = 999.0                                  # rewrite the last word (the "answer")
    out2, _ = attention(later, later, later, causal_mask(4))
    assert torch.allclose(out1[0, :3], out2[0, :3])      # rows 0-2 never saw it


# ---------- Ready for Stage 4 ----------
def test_works_with_head_dimension():
    # [B, heads, S, d_k]; mask [B, 1, 1, S] broadcasts over heads and rows
    src = _src(72, 4072)
    x = torch.randn(2, 8, 4, 16)
    out, w = attention(x, x, x, pad_mask(src).unsqueeze(1))
    assert out.shape == (2, 8, 4, 16)
    assert w.shape == (2, 8, 4, 4)
    assert (w[0, :, :, 2:] == 0).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
def test_runs_on_gpu():
    x = torch.randn(2, 4, 16, device="cuda")
    out, _ = attention(x, x, x, causal_mask(4, device="cuda"))
    assert out.device.type == "cuda"