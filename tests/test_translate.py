import os

import pytest
import torch

from data import EOS_ID, SOS_ID, SRC_VOCAB, TGT_VOCAB
from model import Transformer
from model_training import Checkpoint_path, load_checkpoint
from translate import (beam_decode, encode_number, encode_numbers, greedy_decode, greedy_exact,
                       greedy_translate, ids_to_words, length_penalty)


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


def _model():
    return Transformer(src_vocab_size=len(SRC_VOCAB), tgt_vocab_size=len(TGT_VOCAB))


def _always_eos_model():
    # bias trick: <eos> (ID 2) gets +100, so the model says <eos> immediately -> [1, 2]
    model = _model()
    with torch.no_grad():
        model.out.bias[EOS_ID] = 100.0
    return model


# ---------- Helpers ----------
def test_encode_number_4072():
    assert encode_number(4072).tolist() == [[7, 3, 10, 5]]                     # "0" = 3 ... "9" = 12


def test_encode_numbers_pads_short_cards():
    # Example: 72 and 4072 in one batch, 72 is padded with two zeros
    assert encode_numbers([72, 4072]).tolist() == [[10, 5, 0, 0], [7, 3, 10, 5]]


def test_ids_to_words_4072():
    assert ids_to_words([1, 11, 27, 20, 31, 2]) == "four thousand seventy two"
    assert ids_to_words([1, 20, 31, 2, 0, 0]) == "seventy two"                 # stops at <eos>, pads ignored


# ---------- Greedy ----------
def test_greedy_starts_with_sos_and_respects_max_len():
    # untrained model may never say <eos>, so it must stop at 11 tokens
    ids = greedy_decode(_model(), encode_numbers([72, 4072]))
    assert ids.size(0) == 2
    assert ids.size(1) <= 11
    assert (ids[:, 0] == SOS_ID).all()


def test_greedy_stops_at_eos():
    ids = greedy_decode(_always_eos_model(), encode_numbers([72, 4072]))
    assert ids.tolist() == [[1, 2], [1, 2]]                                    # both done after 1 step


def test_batched_greedy_matches_one_card_at_a_time():
    # Example: 72 and 4072, padding in the batch must not change 72's answer
    model = _model()
    batch = greedy_decode(model, encode_numbers([72, 4072]))
    for row, number in zip(batch.tolist(), [72, 4072]):
        alone = greedy_decode(model, encode_number(number))[0].tolist()
        assert ids_to_words(row) == ids_to_words(alone)


def test_greedy_exact_counts_wrong_cards():
    # always-<eos> model writes "" for every card, so every card is wrong
    accuracy, wrong = greedy_exact(_always_eos_model(), [72, 4072], batch_size=128)
    assert accuracy == 0.0
    assert wrong == [(72, "seventy two", ""), (4072, "four thousand seventy two", "")]


# ---------- Beam search ----------
def test_length_penalty_values():
    assert length_penalty(1) == pytest.approx(1.0)                             # (5 + 1) / 6 = 1
    assert length_penalty(5) == pytest.approx((10 / 6) ** 0.6)                # about 1.36


def test_beam_stops_at_eos():
    assert beam_decode(_always_eos_model(), encode_number(4072)) == [1, 2]


def test_beam_respects_max_len():
    ids = beam_decode(_model(), encode_number(4072), beam_size=4)
    assert ids[0] == SOS_ID
    assert len(ids) <= 11


def test_beam_size_1_equals_greedy():
    # with only 1 draft kept, beam search IS greedy
    model = _model()
    greedy = greedy_decode(model, encode_number(4072))[0].tolist()
    beam = beam_decode(model, encode_number(4072), beam_size=1)
    assert beam == greedy


# ---------- Trained model (skipped if checkpoints/best.pt is missing, e.g. in CI) ----------
@pytest.mark.skipif(not os.path.exists(Checkpoint_path), reason="no trained checkpoint")
def test_trained_model_translates_4072():
    model, _ = load_checkpoint(path=Checkpoint_path, device="cpu")
    assert greedy_translate(model, number=4072, device="cpu") == "four thousand seventy two"