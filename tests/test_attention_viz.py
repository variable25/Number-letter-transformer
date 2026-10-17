import os

import pytest
import torch

from data import EOS_ID, SRC_VOCAB, TGT_VOCAB
from model import Transformer
from model_training import Checkpoint_path, load_checkpoint
from attention_viz import (collect_weights, map_labels, plot_average, plot_heads, summarize_heads,
                           translate_and_collect)

# Example: 4072 with its TRUE answer (real data), digits [7, 3, 10, 5], words <sos> four thousand seventy two
SRC_4072 = torch.tensor([[7, 3, 10, 5]])
TGT_IN_4072 = torch.tensor([[1, 11, 27, 20, 31]])


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


# ---------- Collect ----------
def test_collect_shapes_4072():
    weights = collect_weights(_model(), src=SRC_4072, tgt_in=TGT_IN_4072)
    assert weights["encoder_self"].shape == (3, 8, 4, 4)                     # digits x digits
    assert weights["decoder_self"].shape == (3, 8, 5, 5)                     # words x words
    assert weights["cross"].shape == (3, 8, 5, 4)                            # words x digits


def test_every_row_sums_to_one():
    weights = collect_weights(_model(), src=SRC_4072, tgt_in=TGT_IN_4072)
    for kind, grids in weights.items():
        row_sums = grids.sum(dim=-1)                                         # add up each row
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), kind


def test_decoder_self_is_causal():
    # covering exam answers: every cell above the diagonal must be exactly 0
    weights = collect_weights(_model(), src=SRC_4072, tgt_in=TGT_IN_4072)
    above_diagonal = torch.triu(weights["decoder_self"], diagonal=1)
    assert (above_diagonal == 0).all()


def test_whiteboard_is_overwritten():
    # Example: 72 first (2 digits, 3 words), then 4072: only 4072's shapes may be left
    model = _model()
    collect_weights(model, src=torch.tensor([[10, 5]]), tgt_in=torch.tensor([[1, 20, 31]]))
    weights = collect_weights(model, src=SRC_4072, tgt_in=TGT_IN_4072)
    assert weights["cross"].shape == (3, 8, 5, 4)


def test_labels_with_always_eos_model():
    # the model writes nothing: 1 row (<eos> being written), page holds only <sos>
    result = translate_and_collect(_always_eos_model(), number=4072)
    assert result["words"] == ""
    assert result["digits"] == ["4", "0", "7", "2"]
    assert result["page"] == ["<sos>"]
    assert result["written"] == ["<eos>"]
    assert map_labels(result, kind="cross") == (["<eos>"], ["4", "0", "7", "2"])
    assert result["weights"]["cross"].shape == (3, 8, 1, 4)


# ---------- Compare heads ----------
def test_summarize_heads_picks_biggest_weight():
    # Example: made-up toy grid for 72 (NOT model output), 1 layer, 1 head
    # rows = seventy, two, <eos>   cols = digits 7, 2
    grids = torch.tensor([[[[0.9, 0.1],
                            [0.2, 0.8],
                            [0.3, 0.7]]]])                                   # [1, 1, 3, 2]
    lines = summarize_heads(grids=grids, row_labels=["seventy", "two", "<eos>"], col_labels=["7", "2"])
    assert lines == ["layer 1 head 1 | seventy->7 0.90 | two->2 0.80 | <eos>->2 0.70"]


# ---------- Pictures (tmp_path = a throwaway folder pytest deletes for us) ----------
def test_plot_heads_writes_png(tmp_path):
    weights = collect_weights(_model(), src=SRC_4072, tgt_in=TGT_IN_4072)
    path = plot_heads(grids=weights["cross"],
                      row_labels=["four", "thousand", "seventy", "two", "<eos>"],
                      col_labels=["4", "0", "7", "2"],
                      title="test", path=os.path.join(tmp_path, "cross_attention.png"))
    assert os.path.getsize(path) > 0


def test_plot_average_writes_png(tmp_path):
    weights = collect_weights(_model(), src=SRC_4072, tgt_in=TGT_IN_4072)
    path = plot_average(grid=weights["cross"][-1].mean(dim=0),
                        row_labels=["four", "thousand", "seventy", "two", "<eos>"],
                        col_labels=["4", "0", "7", "2"],
                        title="test", path=os.path.join(tmp_path, "cross_attention_average.png"))
    assert os.path.getsize(path) > 0


# ---------- Trained model (skipped if checkpoints/best.pt is missing, e.g. in CI) ----------
@pytest.mark.skipif(not os.path.exists(Checkpoint_path), reason="no trained checkpoint")
def test_trained_model_4072_maps():
    model, _ = load_checkpoint(path=Checkpoint_path, device="cpu")
    result = translate_and_collect(model, number=4072, device="cpu")
    assert result["words"] == "four thousand seventy two"
    assert result["written"] == ["four", "thousand", "seventy", "two", "<eos>"]
    assert result["weights"]["cross"].shape == (3, 8, 5, 4)