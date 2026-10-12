import pytest
import torch
from torch import nn

from data import SRC_VOCAB, NumberDataset, collate, src_tokens
from embeddings import Embedder
from encoder import D_ff, N_layers, Encoder, EncoderLayer, FeedForward


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


def _src(*numbers):
    return collate([NumberDataset([n])[0] for n in numbers])[0]


def _ids_4072():
    # Example: 4072, real data, our running example. "4" "0" "7" "2" -> IDs [[7, 3, 10, 5]]
    return torch.tensor([SRC_VOCAB.encode(src_tokens(4072))])


def _x_4072():
    # 4072 after the Stage 2 Embedder: [1, 4, 128]
    return Embedder(len(SRC_VOCAB)).eval()(_ids_4072())


def _layer_norm(v):
    # LayerNorm written out by hand, with gamma = 1 and beta = 0 (their values before training)
    mean = v.mean(-1, keepdim=True)                          # average of each digit's 128 numbers
    var = v.var(-1, keepdim=True, unbiased=False)            # spread of each digit's 128 numbers
    return (v - mean) / torch.sqrt(var + 1e-5)


# ---------- Feed-forward (section 3.3) ----------
def test_feed_forward_sizes():
    ffn = FeedForward()
    assert D_ff == 512
    assert ffn.stretch.weight.shape == (512, 128) and ffn.stretch.bias.shape == (512,)
    assert ffn.squeeze.weight.shape == (128, 512) and ffn.squeeze.bias.shape == (128,)
    assert sum(p.numel() for p in ffn.parameters()) == 131_712   # 65,536 + 512 + 65,536 + 128


def test_feed_forward_matches_formula():
    # FFN(x) = max(0, x W1 + b1) W2 + b2
    x = _x_4072()
    ffn = FeedForward()
    hidden = (x @ ffn.stretch.weight.T + ffn.stretch.bias).clamp(min=0)   # [1, 4, 512], negatives -> 0
    expected = hidden @ ffn.squeeze.weight.T + ffn.squeeze.bias            # [1, 4, 128]
    assert torch.allclose(ffn(x), expected, atol=1e-4)


def test_feed_forward_is_position_wise():
    # the "7" (position 2 of 4072) gets the same answer alone as it does next to "4", "0", "2"
    x = _x_4072()
    ffn = FeedForward()
    together = ffn(x)[0, 2]                                  # "7" processed inside 4072
    alone = ffn(x[:, 2:3])[0, 0]                             # "7" processed by itself
    assert torch.allclose(together, alone, atol=1e-4)


# ---------- One encoder layer (Add & Norm) ----------
def test_encoder_layer_sizes():
    layer = EncoderLayer()
    # attention 65,536 + feed-forward 131,712 + 2 LayerNorms x (128 gamma + 128 beta)
    assert sum(p.numel() for p in layer.parameters()) == 65_536 + 131_712 + 2 * 256
    assert layer(_x_4072()).shape == (1, 4, 128)             # same shape in and out


def test_encoder_layer_matches_paper_recipe():
    # LayerNorm(x + Sublayer(x)), twice. eval() turns dropout off so we can compare exactly
    x = _x_4072()
    layer = EncoderLayer().eval()
    h = _layer_norm(x + layer.self_attn(x, x, x))            # sublayer 1: attention, Add & Norm
    expected = _layer_norm(h + layer.ffn(h))                 # sublayer 2: feed-forward, Add & Norm
    assert torch.allclose(layer(x), expected, atol=1e-4)


def test_every_digit_comes_out_normalized():
    # after the last Norm, each digit's 128 numbers have average 0 and spread 1
    out = EncoderLayer().eval()(_x_4072())
    assert torch.allclose(out.mean(-1), torch.zeros(1, 4), atol=1e-5)
    assert torch.allclose(out.var(-1, unbiased=False), torch.ones(1, 4), atol=1e-3)


def test_residual_carries_x_when_sublayers_are_silent():
    # force both sublayers to output all zeros: the skip arrows still carry x through
    x = _x_4072()
    layer = EncoderLayer().eval()
    with torch.no_grad():
        layer.self_attn.W_O.weight.zero_()                   # attention now outputs 0
        layer.ffn.squeeze.weight.zero_()                     # feed-forward now outputs 0
        layer.ffn.squeeze.bias.zero_()
    assert torch.allclose(layer(x), _layer_norm(x), atol=1e-4)


# ---------- The full encoder (section 3.1) ----------
def test_encoder_sizes():
    enc = Encoder(len(SRC_VOCAB))
    assert N_layers == 3
    assert isinstance(enc.layers, nn.ModuleList) and len(enc.layers) == 3
    # 13 x 128 embedding table + 3 layers x 197,760
    assert sum(p.numel() for p in enc.parameters()) == 13 * 128 + 3 * 197_760
    # 3 separate layers with their own weights, not one layer reused 3 times
    assert enc.layers[0] is not enc.layers[1]
    assert not torch.equal(enc.layers[0].ffn.stretch.weight, enc.layers[1].ffn.stretch.weight)


def test_4072_becomes_memory():
    enc = Encoder(len(SRC_VOCAB)).eval()
    memory = enc(_ids_4072())
    assert memory.shape == (1, 4, 128)                       # one understood row per digit
    for layer in enc.layers:
        assert layer.self_attn.weights.shape == (1, 8, 4, 4) # every layer: 8 heads, 4 digits x 4 digits


def test_pads_cannot_change_real_digits():
    # Example: 72 alone ([10, 5]) vs 72 batched with 4072 (padded to [10, 5, 0, 0])
    enc = Encoder(len(SRC_VOCAB)).eval()
    alone = enc(_src(72))                                    # [1, 2, 128]
    batch = enc(_src(72, 4072))                              # [2, 4, 128]
    assert torch.allclose(batch[0, :2], alone[0], atol=1e-4)
    for layer in enc.layers:
        assert (layer.self_attn.weights[0, :, :, 2:] == 0).all()   # no head in any layer looks at 72's pads


def test_dropout_only_while_training():
    enc = Encoder(len(SRC_VOCAB))
    ids = _ids_4072()
    enc.train()
    assert not torch.equal(enc(ids), enc(ids))               # different random zeros each run
    enc.eval()
    assert torch.equal(enc(ids), enc(ids))                   # no dropout: same answer every time


def test_every_weight_can_be_nudged():
    # forward -> loss -> backward: every weight in all 3 layers must receive a gradient
    enc = Encoder(len(SRC_VOCAB))
    enc(_ids_4072()).pow(2).mean().backward()
    assert all(p.grad is not None for p in enc.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
def test_runs_on_gpu():
    enc = Encoder(len(SRC_VOCAB)).to("cuda")
    memory = enc(_src(72, 4072).to("cuda"))
    assert memory.device.type == "cuda" and memory.shape == (2, 4, 128)