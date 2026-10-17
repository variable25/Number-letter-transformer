"""
Training (paper section 5)

One training step on a batch of flashcards:
1. tgt_in = tgt[:, :-1], expected = tgt[:, 1:]       teacher forcing: the decoder sees the true previous words
2. logits = model(src, tgt_in)                        [B, S_tgt, 33]
3. loss = CrossEntropy(logits, expected)              pads ignored, label smoothing 0.1 (5.4)
4. zero_grad -> backward -> clip to norm 1.0 -> Adam step -> learning-rate step (5.3)

After every epoch: validation loss + exact-match accuracy (teacher forcing), save the best checkpoint.
Run once with:  python -c "from train import train; train()"

Batch 4072 + 72: tgt [2, 6] -> tgt_in [2, 5], expected [2, 5] -> logits [2, 5, 33]
"""

import os
import time

import torch
from torch import nn

from data import PAD_ID, SRC_VOCAB, TGT_VOCAB, get_loaders
from embeddings import D_model
from encoder import D_ff, N_layers
from model import Transformer
from multi_head import N_heads

Warmup_steps = 2000                                   # about 5 epochs of 391 batches
N_epochs = 20                                         # 20 x 391 = 7,820 steps
Label_smoothing = 0.1                                 # paper 5.4
Clip_norm = 1.0                                       # not in the paper: our seatbelt
Checkpoint_path = os.path.join("checkpoints", "best.pt")


# ---------- Optimizer and learning-rate schedule (5.3) ----------
def lrate(step, d_model=D_model, warmup_steps=Warmup_steps):
    # straight line up for warmup_steps, then down like 1 / sqrt(step); peak at step == warmup_steps
    return d_model ** -0.5 * min(step ** -0.5, step * warmup_steps ** -1.5)


def make_optimizer(model, d_model=D_model, warmup_steps=Warmup_steps):
    # lr=1.0 is a placeholder: LambdaLR multiplies it by lrate(...) before every step
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    # LambdaLR counts i from 0, the paper counts steps from 1 (0 ** -0.5 would divide by zero)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda i: lrate(i + 1, d_model=d_model, warmup_steps=warmup_steps))
    return optimizer, scheduler


# ---------- Loss (5.4) ----------
def make_criterion(pad_id=PAD_ID, smoothing=Label_smoothing):
    # softmax + cross-entropy in one; <pad> slots are skipped; target is 0.903 on the right word, 0.003 elsewhere
    return nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=smoothing)


def split_target(tgt):
    # tgt [B, S] -> tgt_in [B, S - 1] (last column chopped), expected [B, S - 1] (first column chopped)
    return tgt[:, :-1], tgt[:, 1:]


def compute_loss(model, criterion, src, tgt):
    tgt_in, expected = split_target(tgt)
    logits = model(src, tgt_in)                                                # [B, S, 33]
    # flatten to one long row of word slots; reshape, not view: expected is not contiguous
    loss = criterion(logits.reshape(-1, logits.size(-1)), expected.reshape(-1))   # [B*S, 33] vs [B*S]
    return loss, logits, expected


def exact_match(logits, expected, pad_id=PAD_ID):
    # a card counts only if EVERY real word slot is right; pad slots are forgiven
    preds = logits.argmax(dim=-1)                                              # [B, S]
    ok = (preds == expected) | (expected == pad_id)                            # [B, S] True / False
    return ok.all(dim=1)                                                       # [B] one True / False per card


# ---------- One step, one epoch, validation ----------
def train_step(model, criterion, optimizer, scheduler, src, tgt, clip_norm=Clip_norm):
    model.train()                                                              # dropout on
    loss, _, _ = compute_loss(model, criterion, src, tgt)
    optimizer.zero_grad()                                                      # wipe old .grad
    loss.backward()                                                            # fill .grad for every weight
    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)   # norm BEFORE clipping
    optimizer.step()                                                           # Adam nudges every weight
    scheduler.step()                                                           # new lr for the next step
    return loss.item(), grad_norm.item()


def train_epoch(model, criterion, optimizer, scheduler, loader, device, clip_norm=Clip_norm):
    total_loss, total_words, max_norm = 0.0, 0, 0.0
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        loss, grad_norm = train_step(model=model, criterion=criterion, optimizer=optimizer,
                                     scheduler=scheduler, src=src, tgt=tgt, clip_norm=clip_norm)
        n_words = (tgt[:, 1:] != PAD_ID).sum().item()                          # real (non-pad) expected words
        total_loss += loss * n_words
        total_words += n_words
        max_norm = max(max_norm, grad_norm)
    return total_loss / total_words, max_norm


def evaluate(model, criterion, loader, device):
    model.eval()                                                               # dropout off
    total_loss, total_words, correct_cards, total_cards = 0.0, 0, 0, 0
    with torch.no_grad():                                                      # no gradients: never learns from val
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            loss, logits, expected = compute_loss(model, criterion, src, tgt)
            n_words = (expected != PAD_ID).sum().item()
            total_loss += loss.item() * n_words
            total_words += n_words
            correct_cards += exact_match(logits, expected).sum().item()
            total_cards += src.size(0)
    return total_loss / total_words, correct_cards / total_cards


# ---------- Checkpoints ----------
def make_config():
    # everything needed to rebuild the same Transformer in Stage 9
    return {"src_vocab_size": len(SRC_VOCAB), "tgt_vocab_size": len(TGT_VOCAB), "d_model": D_model,
            "n_heads": N_heads, "d_ff": D_ff, "n_layers": N_layers, "dropout": 0.1,
            "pad_id": PAD_ID, "tie_weights": True}


def save_checkpoint(path, model, config, epoch, val_loss, val_exact):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)                   # create checkpoints/ if missing
    torch.save({"model": model.state_dict(), "config": config, "epoch": epoch,
                "val_loss": val_loss, "val_exact": val_exact}, path)


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = Transformer(**checkpoint["config"]).to(device)                     # same shape, random weights...
    model.load_state_dict(checkpoint["model"])                                 # ...replaced by the saved ones
    model.eval()
    return model, checkpoint


# ---------- The one real run ----------
def train(n_epochs=N_epochs, batch_size=128, warmup_steps=Warmup_steps, checkpoint_path=Checkpoint_path, seed=0):
    torch.manual_seed(seed)                                                    # same start every run
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader, _ = get_loaders(batch_size=batch_size)
    config = make_config()
    model = Transformer(**config).to(device)
    criterion = make_criterion()
    optimizer, scheduler = make_optimizer(model, warmup_steps=warmup_steps)
    n_weights = sum(p.numel() for p in model.parameters())
    print(f"device {device} | {len(train_loader)} batches per epoch | {n_weights:,} weights")

    best_val_loss = float("inf")
    for epoch in range(1, n_epochs + 1):
        start = time.time()
        train_loss, max_norm = train_epoch(model=model, criterion=criterion, optimizer=optimizer,
                                           scheduler=scheduler, loader=train_loader, device=device)
        val_loss, val_exact = evaluate(model=model, criterion=criterion, loader=val_loader, device=device)
        mark = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(path=checkpoint_path, model=model, config=config, epoch=epoch,
                            val_loss=val_loss, val_exact=val_exact)
            mark = "  <- saved"
        print(f"epoch {epoch:2d} | train {train_loss:.3f} | val {val_loss:.3f} | exact {val_exact:.1%}"
              f" | lr {scheduler.get_last_lr()[0]:.5f} | max grad norm {max_norm:.2f}"
              f" | {time.time() - start:.0f}s{mark}")
    return best_val_loss