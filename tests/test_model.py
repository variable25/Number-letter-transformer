import pytest
import torch
from torch import nn

from attention import pad_mask
from data import SRC_VOCAB, TGT_VOCAB, NumberDataset, collate
from decoder import Decoder
from encoder import Encoder
from model import Transformer


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


def _card(*numbers):
    # real flashcards -> (src [B, S_src], tgt [B, S_tgt + 1]), padded exactly like the DataLoader does
    return collate([NumberDataset([n])[0] for n in numbers])


def _72():
    # Example: flashcard 72, real data, shortest interesting card (2 digits, 2 words)
    # src = [[10, 5]]   tgt_in = [[1, 20, 31]] = <sos> seventy two (last column chopped = shifted right)
    src, tgt = _card(72)
    return src, tgt[:, :-1]


def _model(tie_weights=True):
    # 13 digit tokens in, 33 word tokens out
    return Transformer(src_vocab_size=len(SRC_VOCAB), tgt_vocab_size=len(TGT_VOCAB), tie_weights=tie_weights)


# ---------- Building the model (Figure 1, section 3.4) ----------
def test_model_sizes():
    model = _model()
    assert isinstance(model.encoder, Encoder) and isinstance(model.decoder, Decoder)
    assert isinstance(model.out, nn.Linear)
    assert model.out.in_features == 128 and model.out.out_features == 33
    # encoder 594,944 + decoder 794,880 + Linear bias 33 (Linear weight is shared, counted once)
    assert sum(p.numel() for p in model.parameters()) == 594_944 + 794_880 + 33   # 1,389,857


def test_untied_model_sizes():
    model = _model(tie_weights=False)
    assert model.out.weight is not model.decoder.embed.tok.table.weight          # two separate tables
    # encoder + decoder + Linear (33 x 128 weight + 33 bias)
    assert sum(p.numel() for p in model.parameters()) == 594_944 + 794_880 + 33 * 128 + 33   # 1,394,081


def test_tied_weights_are_one_tensor():
    model = _model()
    table = model.decoder.embed.tok.table.weight
    assert model.out.weight is table                                             # same object, not a copy
    assert model.out.weight.shape == (33, 128)                                   # one row per word
    with torch.no_grad():
        model.out.weight[20] += 1.0                                              # nudge the "seventy" row of the Linear...
    assert torch.equal(table[20], model.out.weight[20])                          # ...the embedding row moved too


def test_tied_table_starts_small_with_zero_pad_row():
    table = _model().decoder.embed.tok.table.weight
    assert (table[0] == 0).all()                                                 # <pad> row starts at 0
    assert abs(table[1:].std().item() - 128 ** -0.5) < 0.01                      # spread about 0.088


def test_source_table_is_not_shared():
    # 3-way sharing is impossible: 13 digit rows vs 33 word rows
    model = _model()
    assert model.encoder.embed.tok.table.weight.shape == (13, 128)
    assert model.decoder.embed.tok.table.weight.shape == (33, 128)
    assert model.encoder.embed.tok.table.weight is not model.out.weight


# ---------- One flashcard through the whole model ----------
def test_72_gives_33_scores_per_slot():
    src, tgt_in = _72()
    logits = _model().eval()(src, tgt_in)
    assert logits.shape == (1, 3, 33)                                            # 1 card x 3 slots x 33 words


def test_model_equals_its_parts_by_hand():
    # forward = encoder -> decoder (with pad_mask(src)) -> Linear. eval() turns dropout off
    src, tgt_in = _72()
    model = _model().eval()
    memory = model.encoder(src)                                                  # [1, 2, 128]
    x = model.decoder(tgt_in, memory=memory, src_mask=pad_mask(src))             # [1, 3, 128]
    assert torch.allclose(model(src, tgt_in), model.out(x), atol=1e-6)


def test_score_is_a_dot_product_with_the_word_row():
    # slot 0 (<sos>) score for word 20 (seventy) = x[0, 0] . W[20] + b[20]
    src, tgt_in = _72()
    model = _model().eval()
    x = model.decoder(tgt_in, memory=model.encoder(src), src_mask=pad_mask(src))
    logits = model(src, tgt_in)
    by_hand = x[0, 0] @ model.out.weight[20] + model.out.bias[20]
    assert torch.allclose(logits[0, 0, 20], by_hand, atol=1e-5)


def test_first_scores_are_moderate():
    # thanks to the small tied table, untrained scores have spread around 1, not around 11
    src, tgt_in = _72()
    logits = _model().eval()(src, tgt_in)
    assert logits.std().item() < 3


def test_softmax_gives_probabilities_and_keeps_the_winner():
    src, tgt_in = _72()
    logits = _model().eval()(src, tgt_in)
    probs = logits.softmax(dim=-1)                                               # [1, 3, 33]
    assert torch.allclose(probs.sum(-1), torch.ones(1, 3), atol=1e-5)            # each slot sums to 1
    assert torch.equal(probs.argmax(-1), logits.argmax(-1))                      # same top word per slot


# ---------- Batches, masks, training mode ----------
def test_pads_cannot_change_real_scores():
    # Example: 72 alone vs 72 batched with 4072
    # batched: src [10, 5, 0, 0], tgt_in <sos> seventy two <eos> <pad>
    model = _model().eval()
    src_a, tgt_a = _card(72)
    alone = model(src_a, tgt_a[:, :-1])                                          # [1, 3, 33]
    src_b, tgt_b = _card(72, 4072)
    batch = model(src_b, tgt_b[:, :-1])                                          # [2, 5, 33]
    assert batch.shape == (2, 5, 33)
    assert torch.allclose(batch[0, :3], alone[0], atol=1e-4)


def test_changing_a_future_word_does_not_change_past_scores():
    # Example: decoder inputs of 72 ("<sos> seventy two") and 73 ("<sos> seventy three"), same digits (72's)
    src, tgt_72 = _72()
    _, tgt_73 = _card(73)
    model = _model().eval()
    out_72 = model(src, tgt_72)
    out_73 = model(src, tgt_73[:, :-1])
    assert torch.allclose(out_72[0, :2], out_73[0, :2], atol=1e-5)               # <sos> and seventy: unchanged
    assert not torch.allclose(out_72[0, 2], out_73[0, 2], atol=1e-5)             # last slot: changed


def test_dropout_only_while_training():
    src, tgt_in = _72()
    model = _model()
    model.train()
    assert not torch.equal(model(src, tgt_in), model(src, tgt_in))
    model.eval()
    assert torch.equal(model(src, tgt_in), model(src, tgt_in))


def test_every_weight_can_be_nudged():
    # forward -> loss -> backward: every weight in encoder, decoder and Linear gets a gradient
    src, tgt_in = _72()
    model = _model()
    model(src, tgt_in).pow(2).mean().backward()
    assert all(p.grad is not None for p in model.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
def test_runs_on_gpu():
    model = _model().to("cuda")
    assert model.out.weight is model.decoder.embed.tok.table.weight              # still tied after .to
    src, tgt = _card(72, 4072)
    logits = model(src.to("cuda"), tgt[:, :-1].to("cuda"))
    assert logits.device.type == "cuda" and logits.shape == (2, 5, 33)