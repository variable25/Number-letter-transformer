import pytest
import torch

from data import (EOS_ID, MAX_N, PAD_ID, SOS_ID, SRC_VOCAB, TGT_VOCAB,
                  NumberDataset, collate, make_splits, number_to_words, tgt_tokens)

@pytest.mark.parametrize("n, words", [
    (0, "zero"),
    (7, "seven"),
    (15, "fifteen"),
    (40, "forty"),
    (100, "one hundred"),
    (1000, "one thousand"),
    (4072, "four thousand seventy two"),
    (10_001, "ten thousand one"),
    (999_999, "nine hundred ninety nine thousand nine hundred ninety nine"),
])
def test_number_to_words(n, words):
    assert number_to_words(n) == words


def test_vocab_sizes_and_specials():
    assert len(SRC_VOCAB) == 13
    assert len(TGT_VOCAB) == 33
    for vocab in (SRC_VOCAB, TGT_VOCAB):
        assert vocab.itos[:3] == ["<pad>", "<sos>", "<eos>"]
        assert [vocab.stoi[t] for t in ("<pad>", "<sos>", "<eos>")] == [PAD_ID, SOS_ID, EOS_ID]


def test_every_number_fits_target_vocab():
    # Walk all 1,000,000 numbers: every word must be in the vocab, and every vocab word used.
    seen, longest = set(), 0
    for n in range(MAX_N + 1):
        words = tgt_tokens(n)
        seen.update(words)
        longest = max(longest, len(words))
    assert seen == set(TGT_VOCAB.itos[3:])
    assert longest == 9  # e.g. 777,777 -> 10 words (+ <sos>/<eos> = 12 tokens)


@pytest.mark.parametrize("n", [0, 7, 4072, 602_053, 999_999])
def test_dataset_item_round_trip(n):
    src, tgt = NumberDataset([n])[0]
    assert "".join(SRC_VOCAB.itos[i] for i in src.tolist()) == str(n)
    assert tgt[0] == SOS_ID and tgt[-1] == EOS_ID
    assert " ".join(TGT_VOCAB.decode(tgt.tolist())) == number_to_words(n)


def test_splits_sizes_disjoint_and_cover_small_numbers():
    train, val, test = make_splits()
    assert (len(train), len(val), len(test)) == (50_000, 2_000, 2_000)
    tr, va, te = set(train), set(val), set(test)
    assert len(tr) == len(train)  # no duplicate cards
    assert not (tr & va or tr & te or va & te)
    assert set(range(1000)) <= tr  # every building block is in train
    assert min(val + test) >= 1000


def test_splits_are_reproducible():
    assert make_splits() == make_splits()  # same seed -> same cards


def test_collate_pads_to_longest_in_batch():
    batch = [NumberDataset([n])[0] for n in (7, 4072)]
    src, tgt = collate(batch)
    assert src.tolist() == [[10, PAD_ID, PAD_ID, PAD_ID], [7, 3, 10, 5]]
    assert tgt.shape == (2, 6)  # 4072: <sos> + 4 words + <eos>
    assert tgt[0].tolist() == [SOS_ID, TGT_VOCAB.stoi["seven"], EOS_ID, PAD_ID, PAD_ID, PAD_ID]
    assert src.dtype == tgt.dtype == torch.long
