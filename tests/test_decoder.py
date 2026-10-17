import pytest
import torch
from torch import nn

from attention import causal_mask, pad_mask
from data import SRC_VOCAB, TGT_VOCAB, NumberDataset, collate
from decoder import Decoder, DecoderLayer
from embeddings import Embedder
from encoder import Encoder


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


def _card(*numbers):
    # real flashcards -> (src [B, S_src], tgt [B, S_tgt + 1]), padded exactly like the DataLoader does
    return collate([NumberDataset([n])[0] for n in numbers])


def _72():
    # Example: flashcard 72, real data, shortest interesting card (2 digits, 2 words)
    # src = [[10, 5]]   tgt_in = <sos> seventy two   (tgt with its last column chopped = shifted right)
    src, tgt = _card(72)
    return src, tgt[:, :-1]


def _layer_norm(v):
    # LayerNorm written out by hand, with gamma = 1 and beta = 0 (their values before training)
    mean = v.mean(-1, keepdim=True)
    var = v.var(-1, keepdim=True, unbiased=False)
    return (v - mean) / torch.sqrt(var + 1e-5)


def _layer_inputs_72():
    # everything one DecoderLayer needs for flashcard 72
    src, tgt_in = _72()
    x = Embedder(len(TGT_VOCAB)).eval()(tgt_in)                          # words  [1, 3, 128]
    memory = Encoder(len(SRC_VOCAB)).eval()(src)                         # digits [1, 2, 128]
    src_mask = pad_mask(src)                                             # [1, 1, 2] = [T, T]
    tgt_mask = pad_mask(tgt_in) & causal_mask(3)                         # [1, 3, 3] lower triangle
    return x, memory, src_mask, tgt_mask


# ---------- One decoder layer (section 3.1) ----------
def test_decoder_layer_sizes():
    layer = DecoderLayer()
    # 2 attentions x 65,536 + feed-forward 131,712 + 3 LayerNorms x (128 gamma + 128 beta)
    assert sum(p.numel() for p in layer.parameters()) == 2 * 65_536 + 131_712 + 3 * 256   # 263,552
    assert layer.self_attn is not layer.cross_attn                       # two separate bricks
    x, memory, src_mask, tgt_mask = _layer_inputs_72()
    out = layer(x, memory=memory, src_mask=src_mask, tgt_mask=tgt_mask)
    assert out.shape == (1, 3, 128)                                      # one row per word slot


def test_decoder_layer_matches_paper_recipe():
    # LayerNorm(x + Sublayer(x)), three times. eval() turns dropout off so we can compare exactly
    x, memory, src_mask, tgt_mask = _layer_inputs_72()
    layer = DecoderLayer().eval()
    h1 = _layer_norm(x + layer.self_attn(x, x, x, tgt_mask))                   # words look back
    h2 = _layer_norm(h1 + layer.cross_attn(h1, memory, memory, src_mask))      # words search digits
    expected = _layer_norm(h2 + layer.ffn(h2))                                 # each word thinks alone
    out = layer(x, memory=memory, src_mask=src_mask, tgt_mask=tgt_mask)
    assert torch.allclose(out, expected, atol=1e-4)


def test_every_word_comes_out_normalized():
    # after norm3, each word slot's 128 numbers have average 0 and spread 1
    x, memory, src_mask, tgt_mask = _layer_inputs_72()
    out = DecoderLayer().eval()(x, memory=memory, src_mask=src_mask, tgt_mask=tgt_mask)
    assert torch.allclose(out.mean(-1), torch.zeros(1, 3), atol=1e-5)
    assert torch.allclose(out.var(-1, unbiased=False), torch.ones(1, 3), atol=1e-3)


def test_residual_carries_x_when_sublayers_are_silent():
    # force all 3 sublayers to output zeros: the skip arrows still carry x through 3 Norms
    x, memory, src_mask, tgt_mask = _layer_inputs_72()
    layer = DecoderLayer().eval()
    with torch.no_grad():
        layer.self_attn.W_O.weight.zero_()
        layer.cross_attn.W_O.weight.zero_()
        layer.ffn.squeeze.weight.zero_()
        layer.ffn.squeeze.bias.zero_()
    out = layer(x, memory=memory, src_mask=src_mask, tgt_mask=tgt_mask)
    assert torch.allclose(out, _layer_norm(_layer_norm(_layer_norm(x))), atol=1e-4)


# ---------- The full decoder (section 3.1 + 3.2.3) ----------
def test_decoder_sizes():
    dec = Decoder(len(TGT_VOCAB))
    assert isinstance(dec.layers, nn.ModuleList) and len(dec.layers) == 3
    assert dec.embed.tok.table.weight.shape == (33, 128)                 # one row per word
    # 33 x 128 embedding table + 3 layers x 263,552
    assert sum(p.numel() for p in dec.parameters()) == 33 * 128 + 3 * 263_552   # 794,880
    assert dec.layers[0] is not dec.layers[1]                            # 3 separate layers


def test_72_writes_one_row_per_word():
    src, tgt_in = _72()
    memory = Encoder(len(SRC_VOCAB)).eval()(src)
    dec = Decoder(len(TGT_VOCAB)).eval()
    out = dec(tgt_in, memory, pad_mask(src))
    assert out.shape == (1, 3, 128)                                      # <sos> seventy two -> 3 rows
    for layer in dec.layers:
        assert layer.self_attn.weights.shape == (1, 8, 3, 3)             # 3 words x 3 words
        assert layer.cross_attn.weights.shape == (1, 8, 3, 2)            # 3 words x 2 digits


def test_future_words_are_covered():
    # covering the exam answers: no head in any layer looks at a later word
    src, tgt_in = _72()
    memory = Encoder(len(SRC_VOCAB)).eval()(src)
    dec = Decoder(len(TGT_VOCAB)).eval()
    dec(tgt_in, memory, pad_mask(src))
    future = torch.triu(torch.ones(3, 3, dtype=torch.bool), diagonal=1)  # True above the diagonal = later words
    for layer in dec.layers:
        assert (layer.self_attn.weights[0][:, future] == 0).all()


def test_changing_a_future_word_does_not_change_the_past():
    # Example: decoder inputs of 72 ("<sos> seventy two") and 73 ("<sos> seventy three"), both real cards.
    # Same memory (72's digits) so only the last word differs.
    src, tgt_72 = _72()
    _, tgt_73 = _card(73)
    tgt_73 = tgt_73[:, :-1]
    memory = Encoder(len(SRC_VOCAB)).eval()(src)
    dec = Decoder(len(TGT_VOCAB)).eval()
    out_72 = dec(tgt_72, memory, pad_mask(src))
    out_73 = dec(tgt_73, memory, pad_mask(src))
    assert torch.allclose(out_72[0, :2], out_73[0, :2], atol=1e-5)       # <sos> and seventy: unchanged
    assert not torch.allclose(out_72[0, 2], out_73[0, 2], atol=1e-5)     # last slot: changed


def test_pads_cannot_change_real_words():
    # Example: 72 alone vs 72 batched with 4072
    # batched: src [10, 5, 0, 0], tgt_in <sos> seventy two <eos> <pad>
    enc = Encoder(len(SRC_VOCAB)).eval()
    dec = Decoder(len(TGT_VOCAB)).eval()
    src_a, tgt_a = _card(72)
    alone = dec(tgt_a[:, :-1], enc(src_a), pad_mask(src_a))             # [1, 3, 128]
    src_b, tgt_b = _card(72, 4072)
    batch = dec(tgt_b[:, :-1], enc(src_b), pad_mask(src_b))             # [2, 5, 128]
    assert torch.allclose(batch[0, :3], alone[0], atol=1e-4)
    for layer in dec.layers:
        assert (layer.cross_attn.weights[0, :, :, 2:] == 0).all()        # no word looks at 72's pad digits
        assert (layer.self_attn.weights[0, :, :, 4] == 0).all()          # no word looks at 72's pad word


def test_dropout_only_while_training():
    src, tgt_in = _72()
    memory = Encoder(len(SRC_VOCAB)).eval()(src)
    dec = Decoder(len(TGT_VOCAB))
    dec.train()
    assert not torch.equal(dec(tgt_in, memory, pad_mask(src)), dec(tgt_in, memory, pad_mask(src)))
    dec.eval()
    assert torch.equal(dec(tgt_in, memory, pad_mask(src)), dec(tgt_in, memory, pad_mask(src)))


def test_every_weight_can_be_nudged():
    # forward -> loss -> backward: every decoder weight gets a gradient,
    # AND the error flows back through memory into every encoder weight
    src, tgt_in = _72()
    enc = Encoder(len(SRC_VOCAB))
    dec = Decoder(len(TGT_VOCAB))
    dec(tgt_in, enc(src), pad_mask(src)).pow(2).mean().backward()
    assert all(p.grad is not None for p in dec.parameters())
    assert all(p.grad is not None for p in enc.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
def test_runs_on_gpu():
    enc = Encoder(len(SRC_VOCAB)).to("cuda")
    dec = Decoder(len(TGT_VOCAB)).to("cuda")
    src, tgt = _card(72, 4072)
    src, tgt_in = src.to("cuda"), tgt[:, :-1].to("cuda")
    out = dec(tgt_in, enc(src), pad_mask(src))
    assert out.device.type == "cuda" and out.shape == (2, 5, 128)