import math

import pytest
import torch
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader

from data import SRC_VOCAB, TGT_VOCAB, NumberDataset, collate
from model import Transformer
from model_training import (compute_loss, evaluate, exact_match, load_checkpoint, lrate, make_config,
                   make_criterion, make_optimizer, save_checkpoint, split_target, train_step)


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


def _card(*numbers):
    # real flashcards -> (src [B, S_src], tgt [B, S_tgt]), padded exactly like the DataLoader does
    return collate([NumberDataset([n])[0] for n in numbers])


def _model():
    return Transformer(src_vocab_size=len(SRC_VOCAB), tgt_vocab_size=len(TGT_VOCAB))


# ---------- Learning-rate schedule and Adam (5.3) ----------
def test_lrate_matches_paper_formula():
    assert lrate(1) == pytest.approx(128 ** -0.5 * 2000 ** -1.5)                  # 0.000001: tiny first step
    assert lrate(2000) == pytest.approx(0.001976, abs=1e-6)                       # peak
    assert lrate(7820) == pytest.approx(0.001000, abs=1e-6)                       # end of epoch 20


def test_lrate_peaks_at_warmup():
    values = [lrate(step) for step in range(1, 7821)]
    assert values.index(max(values)) + 1 == 2000                                  # list index 0 = step 1
    assert lrate(1000) == pytest.approx(lrate(2000) / 2)                          # straight line up
    assert lrate(8000) == pytest.approx(lrate(2000) / 2)                          # 1/sqrt down: 4x steps -> half


def test_optimizer_settings_and_first_lr():
    optimizer, _ = make_optimizer(_model())
    group = optimizer.param_groups[0]
    assert group["betas"] == (0.9, 0.98) and group["eps"] == 1e-9
    assert group["lr"] == pytest.approx(lrate(1))                                 # schedule already applied


def test_scheduler_follows_lrate():
    optimizer, scheduler = make_optimizer(_model(), warmup_steps=10)
    for _ in range(3):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(lrate(4, warmup_steps=10))


# ---------- Teacher forcing and loss (5.4) ----------
def test_split_target_4072_and_72():
    # 4072: <sos> four thousand seventy two <eos>   72: <sos> seventy two <eos> <pad> <pad>
    _, tgt = _card(4072, 72)
    tgt_in, expected = split_target(tgt)
    assert tgt_in.tolist() == [[1, 11, 27, 20, 31], [1, 20, 31, 2, 0]]
    assert expected.tolist() == [[11, 27, 20, 31, 2], [20, 31, 2, 0, 0]]


def test_expected_needs_reshape_not_view():
    _, tgt = _card(4072, 72)
    _, expected = split_target(tgt)
    assert not expected.is_contiguous()                                           # slice skips a column
    with pytest.raises(RuntimeError):
        expected.view(-1)                                                         # .view crashes...
    assert expected.reshape(-1).tolist() == [11, 27, 20, 31, 2, 20, 31, 2, 0, 0]  # ...reshape works


def test_untrained_loss_is_near_random_guessing():
    src, tgt = _card(4072, 72)
    _, expected = split_target(tgt)
    criterion = make_criterion()
    # pure guessing: all 33 scores equal -> exactly ln(33) = 3.50, smoothing or not
    all_equal = torch.zeros(10, 33)                                               # 10 slots of 4072 + 72
    assert criterion(all_equal, expected.reshape(-1)).item() == pytest.approx(math.log(33))
    # untrained model: random scores = a noisy guesser, never much better than ln(33), and not wildly worse
    model = _model().eval()
    with torch.no_grad():
        loss, logits, _ = compute_loss(model, criterion, src, tgt)
    assert logits.shape == (2, 5, 33)
    assert math.log(33) - 0.5 < loss.item() < 8.0


def test_pad_slots_are_ignored_by_loss():
    src, tgt = _card(4072, 72)
    tgt_in, expected = split_target(tgt)
    criterion = make_criterion()
    with torch.no_grad():
        logits = _model().eval()(src, tgt_in)                                     # [2, 5, 33]
        loss = criterion(logits.reshape(-1, 33), expected.reshape(-1))
        changed = logits.clone()
        changed[1, 3:] += 100.0                                                   # 72's two <pad> slots go wild
        loss_changed = criterion(changed.reshape(-1, 33), expected.reshape(-1))
    assert torch.allclose(loss, loss_changed)


def test_label_smoothing_floor():
    # 72's expected words: seventy two <eos>
    expected = torch.tensor([20, 31, 2])
    target = 0.9 * one_hot(expected, 33).float() + 0.1 / 33                       # 0.903 right, 0.003 others
    criterion = make_criterion()
    assert criterion(target.log(), expected).item() == pytest.approx(0.654, abs=0.001)   # best possible
    sure = one_hot(expected, 33).float() * 100.0                                  # 100% sure of the right word
    assert criterion(sure, expected).item() > 5                                   # smoothing punishes it


def test_exact_match_needs_the_whole_card():
    _, tgt = _card(4072, 72)
    _, expected = split_target(tgt)
    preds = expected.clone()
    preds[1, 3:] = torch.tensor([14, 9])                                          # 72's pad slots: nine, five
    assert exact_match(one_hot(preds, 33).float(), expected).tolist() == [True, True]
    preds[0, 3] = 28                                                              # 4072: three instead of two
    assert exact_match(one_hot(preds, 33).float(), expected).tolist() == [False, True]


# ---------- Training step ----------
def test_train_step_lowers_loss():
    src, tgt = _card(4072, 72)
    model = _model()
    criterion = make_criterion()
    optimizer, scheduler = make_optimizer(model, warmup_steps=50)                 # short warmup for a tiny test
    model.eval()
    with torch.no_grad():
        before = compute_loss(model, criterion, src, tgt)[0].item()
    for _ in range(40):
        train_step(model=model, criterion=criterion, optimizer=optimizer, scheduler=scheduler, src=src, tgt=tgt)
    model.eval()
    with torch.no_grad():
        after = compute_loss(model, criterion, src, tgt)[0].item()
    assert after < before - 0.5


def test_train_step_clips_gradients():
    src, tgt = _card(4072, 72)
    model = _model()
    optimizer, scheduler = make_optimizer(model)
    _, grad_norm = train_step(model=model, criterion=make_criterion(), optimizer=optimizer,
                              scheduler=scheduler, src=src, tgt=tgt, clip_norm=1.0)
    total = torch.stack([p.grad.norm() for p in model.parameters()]).norm().item()   # length after clipping
    assert total <= 1.0 + 1e-4
    assert total == pytest.approx(min(grad_norm, 1.0), rel=1e-3)


def test_train_step_moves_schedule():
    src, tgt = _card(4072, 72)
    model = _model()
    optimizer, scheduler = make_optimizer(model)
    loss, _ = train_step(model=model, criterion=make_criterion(), optimizer=optimizer,
                         scheduler=scheduler, src=src, tgt=tgt)
    assert isinstance(loss, float)
    assert model.training                                                         # dropout was on
    assert optimizer.param_groups[0]["lr"] == pytest.approx(lrate(2))             # ready for step 2


# ---------- Validation and checkpoints ----------
def test_evaluate_weights_by_words():
    # 3 real cards in batches of 2: [4072, 72] (5 + 3 words) then [73] (3 words)
    model = _model()
    criterion = make_criterion()
    loader = DataLoader(NumberDataset([4072, 72, 73]), batch_size=2, collate_fn=collate)
    val_loss, val_exact = evaluate(model=model, criterion=criterion, loader=loader, device="cpu")
    assert not model.training                                                     # dropout off
    src, tgt = _card(4072, 72, 73)
    with torch.no_grad():
        one_batch = compute_loss(model, criterion, src, tgt)[0].item()            # all 11 words at once
    assert val_loss == pytest.approx(one_batch, abs=1e-4)
    assert 0.0 <= val_exact <= 1.0


def test_checkpoint_round_trip(tmp_path):
    model = _model().eval()
    path = tmp_path / "checkpoints" / "best.pt"
    save_checkpoint(path=path, model=model, config=make_config(), epoch=7, val_loss=0.68, val_exact=0.97)
    loaded, checkpoint = load_checkpoint(path)
    assert checkpoint["epoch"] == 7 and checkpoint["val_loss"] == 0.68
    assert loaded.out.weight is loaded.decoder.embed.tok.table.weight             # still tied
    src, tgt = _card(72)
    tgt_in, _ = split_target(tgt)
    with torch.no_grad():
        assert torch.allclose(model(src, tgt_in), loaded(src, tgt_in), atol=1e-6)


def test_config_rebuilds_same_model():
    model = Transformer(**make_config())
    assert sum(p.numel() for p in model.parameters()) == 1_389_857


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
def test_train_step_on_gpu():
    model = _model().to("cuda")
    optimizer, scheduler = make_optimizer(model)
    src, tgt = _card(4072, 72)
    loss, grad_norm = train_step(model=model, criterion=make_criterion(), optimizer=optimizer,
                                 scheduler=scheduler, src=src.to("cuda"), tgt=tgt.to("cuda"))
    assert math.isfinite(loss) and grad_norm > 0