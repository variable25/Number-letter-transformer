"""
Translate (paper section 6.1): the trained model writes the words itself

Greedy, one card (4072):
1. src = [[7, 3, 10, 5]]                        the digits, shape [1, 4]
2. memory = encoder(src)                        built ONCE, reused every step
3. tgt_in = [[1]]                               start with <sos>
4. repeat: decoder -> last row -> Linear -> argmax -> glue the new word onto tgt_in
5. stop at <eos> (2) or at 11 tokens            [1, 11, 27, 20, 31, 2] -> "four thousand seventy two"

Batched greedy: same loop for 128 cards at once; cards that already said <eos> get <pad>.
Beam search (6.1): keep the 4 best drafts instead of 1, length penalty alpha = 0.6.
Run once with:  python -c "from translate import report; report()"
"""

import time

import torch
from torch.nn.utils.rnn import pad_sequence

from attention import pad_mask
from data import EOS_ID, PAD_ID, SOS_ID, SRC_VOCAB, TGT_VOCAB, make_splits, number_to_words, src_tokens
from model_training import Checkpoint_path, load_checkpoint

Max_len = 11                                          # longest card: 9 words + <sos> + <eos>
Beam_size = 4                                         # paper 6.1
Length_alpha = 0.6                                    # paper 6.1


# ---------- Small helpers ----------
def encode_number(number, device="cpu"):
    # 4072 -> [[7, 3, 10, 5]]: one card, digit IDs, shape [1, S_src]
    return torch.tensor([SRC_VOCAB.encode(src_tokens(number))], device=device)


def encode_numbers(numbers, device="cpu"):
    # many cards -> [B, S_src], short numbers padded with <pad> = 0 (same as the DataLoader)
    srcs = [torch.tensor(SRC_VOCAB.encode(src_tokens(n))) for n in numbers]
    return pad_sequence(srcs, batch_first=True, padding_value=PAD_ID).to(device)


def ids_to_words(ids):
    # [1, 11, 27, 20, 31, 2] -> "four thousand seventy two" (skips <sos>/<pad>, stops at <eos>)
    return " ".join(TGT_VOCAB.decode(ids))


# ---------- Greedy (batched) ----------
def greedy_decode(model, src, max_len=Max_len, pad_id=PAD_ID, sos_id=SOS_ID, eos_id=EOS_ID):
    # src [B, S_src] -> word IDs [B, L], L <= max_len, every row starts with <sos>
    model.eval()                                                               # dropout off
    with torch.no_grad():                                                      # only reading, never learning
        src_mask = pad_mask(src, pad_id)                                       # [B, 1, S_src]
        memory = model.encoder(src)                                            # [B, S_src, 128], ONCE
        batch_size = src.size(0)
        tgt_in = torch.full((batch_size, 1), sos_id, dtype=torch.long, device=src.device)   # [B, 1] all <sos>
        finished = torch.zeros(batch_size, dtype=torch.bool, device=src.device)            # [B] all False

        while tgt_in.size(1) < max_len:
            x = model.decoder(tgt_in, memory=memory, src_mask=src_mask)        # [B, L, 128]
            next_logits = model.out(x[:, -1, :])                               # last row only -> [B, 33]
            next_ids = next_logits.argmax(dim=-1)                              # [B] best word per card
            next_ids = next_ids.masked_fill(finished, pad_id)                  # done cards get <pad>
            tgt_in = torch.cat([tgt_in, next_ids.unsqueeze(1)], dim=1)         # [B, L] -> [B, L + 1]
            finished = finished | (next_ids == eos_id)                         # said <eos> now -> done
            if finished.all():                                                 # every card done: stop early
                break
    return tgt_in


def greedy_translate(model, number, device="cpu"):
    # 4072 -> "four thousand seventy two"
    ids = greedy_decode(model, encode_number(number, device=device))[0]        # row 0 of [1, L]
    return ids_to_words(ids.tolist())


def greedy_exact(model, numbers, device="cpu", batch_size=128):
    # real exact match: the model's own words vs the true words, card by card
    wrong = []                                                                 # (number, expected, predicted)
    for start in range(0, len(numbers), batch_size):
        chunk = numbers[start:start + batch_size]
        ids = greedy_decode(model, encode_numbers(chunk, device=device))       # [B, L]
        for number, row in zip(chunk, ids.tolist()):
            expected, predicted = number_to_words(number), ids_to_words(row)
            if predicted != expected:
                wrong.append((number, expected, predicted))
    return 1 - len(wrong) / len(numbers), wrong


# ---------- Beam search (6.1, one card at a time) ----------
def length_penalty(length, alpha=Length_alpha):
    # longer answers collect more negative log-probs; dividing by this evens it out
    return ((5 + length) / 6) ** alpha


def beam_decode(model, src, beam_size=Beam_size, max_len=Max_len, alpha=Length_alpha,
                pad_id=PAD_ID, sos_id=SOS_ID, eos_id=EOS_ID):
    # src [1, S_src] -> best list of word IDs, starting with <sos>
    model.eval()
    with torch.no_grad():
        src_mask = pad_mask(src, pad_id)                                       # [1, 1, S_src]
        memory = model.encoder(src)                                            # [1, S_src, 128], ONCE
        beams = [([sos_id], 0.0)]                                              # (draft IDs, total log-prob)
        done = []                                                              # drafts that said <eos>

        while beams and len(beams[0][0]) < max_len:
            tgt_in = torch.tensor([ids for ids, _ in beams], device=src.device)   # [k, L], k = live drafts
            k = tgt_in.size(0)
            x = model.decoder(tgt_in, memory=memory.expand(k, -1, -1),         # same digits for every draft
                              src_mask=src_mask.expand(k, -1, -1))             # [k, L, 128]
            log_probs = torch.log_softmax(model.out(x[:, -1, :]), dim=-1)      # [k, 33], all <= 0

            candidates = []                                                    # every draft + one more word
            for b, (ids, score) in enumerate(beams):
                top_scores, top_ids = log_probs[b].topk(beam_size)             # its best beam_size next words
                for word_score, word_id in zip(top_scores.tolist(), top_ids.tolist()):
                    candidates.append((ids + [word_id], score + word_score))
            candidates.sort(key=lambda c: c[1], reverse=True)                  # best total first

            beams = []
            for ids, score in candidates[:beam_size]:                          # keep only the best beam_size
                if ids[-1] == eos_id:
                    done.append((ids, score))                                  # finished draft, set aside
                else:
                    beams.append((ids, score))                                 # still writing

        done += beams                                                          # drafts cut off at max_len
        # length = words written including <eos> (everything after <sos>)
        best_ids, _ = max(done, key=lambda c: c[1] / length_penalty(len(c[0]) - 1, alpha=alpha))
    return best_ids


def beam_translate(model, number, device="cpu", beam_size=Beam_size):
    return ids_to_words(beam_decode(model, encode_number(number, device=device), beam_size=beam_size))


def beam_exact(model, numbers, device="cpu", beam_size=Beam_size):
    wrong = []
    for number in numbers:
        expected = number_to_words(number)
        predicted = beam_translate(model, number=number, device=device, beam_size=beam_size)
        if predicted != expected:
            wrong.append((number, expected, predicted))
    return 1 - len(wrong) / len(numbers), wrong


# ---------- The one real run ----------
def _show_wrong(wrong, n_show):
    for number, expected, predicted in wrong[:n_show]:
        print(f"  {number:>7} | expected: {expected} | got: {predicted}")


def report(checkpoint_path=Checkpoint_path, batch_size=128, beam_size=Beam_size, n_show=20):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, checkpoint = load_checkpoint(path=checkpoint_path, device=device)
    _, _, test_numbers = make_splits()                                         # the 2,000 test cards
    print(f"checkpoint from epoch {checkpoint['epoch']} | teacher-forced val exact {checkpoint['val_exact']:.1%}")
    print(f"4072 -> {greedy_translate(model, number=4072, device=device)}")

    start = time.time()
    greedy_acc, greedy_wrong = greedy_exact(model, test_numbers, device=device, batch_size=batch_size)
    print(f"greedy   | real exact {greedy_acc:.1%} | wrong {len(greedy_wrong)} of {len(test_numbers)}"
          f" | {time.time() - start:.0f}s")
    _show_wrong(greedy_wrong, n_show)

    start = time.time()
    beam_acc, beam_wrong = beam_exact(model, test_numbers, device=device, beam_size=beam_size)
    print(f"beam {beam_size}   | real exact {beam_acc:.1%} | wrong {len(beam_wrong)} of {len(test_numbers)}"
          f" | {time.time() - start:.0f}s")
    _show_wrong(beam_wrong, n_show)
    return greedy_acc, beam_acc