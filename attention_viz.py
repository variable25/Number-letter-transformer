"""
Attention visualization (paper Figures 3-5: pictures of where the model looks)

One card (4072), trained model:
1. greedy_decode writes the words                  [1, 11, 27, 20, 31, 2]
2. tgt_in = those words without the last one        [1, 11, 27, 20, 31]
3. ONE clean pass model(src, tgt_in): every MultiHeadAttention saves self.weights
   (self.weights is a whiteboard: wiped and rewritten on every call)
4. Collect the 9 bricks, drop B (one card), stack the 3 layers:
      encoder_self  [3, 8, 4, 4]   digits look at digits
      decoder_self  [3, 8, 5, 5]   words look at earlier words (causal triangle)
      cross         [3, 8, 5, 4]   words look at digits
5. Heatmaps -> PNG files in attention_maps/, plus a text summary per head

Run once with:  python -c "from attention_viz import report; report()"
"""

import os

import matplotlib
matplotlib.use("Agg")                                 # draw straight into PNG files, never open a window
import matplotlib.pyplot as plt
import torch

from data import TGT_VOCAB, src_tokens
from model_training import Checkpoint_path, load_checkpoint
from translate import encode_number, greedy_decode, ids_to_words

Out_dir = "attention_maps"
Kinds = ["encoder_self", "decoder_self", "cross"]
Titles = {
    "encoder_self": "Encoder self-attention: digits look at digits",
    "decoder_self": "Decoder self-attention: words look at earlier words",
    "cross": "Cross-attention: words look at digits",
}
# which labels go on (rows, columns) for each kind of map
Map_axes = {
    "encoder_self": ("digits", "digits"),
    "decoder_self": ("written", "page"),
    "cross": ("written", "digits"),
}


# ---------- Collect ----------
def collect_weights(model, src, tgt_in):
    # src [1, S_src], tgt_in [1, S_tgt] -> 3 tensors [3 layers, 8 heads, rows, cols]
    model.eval()                                                               # dropout off
    with torch.no_grad():                                                      # only looking
        model(src, tgt_in)                                                     # one pass fills every whiteboard
    encoder_self = [layer.self_attn.weights[0] for layer in model.encoder.layers]    # [0] drops B: [8, S_src, S_src]
    decoder_self = [layer.self_attn.weights[0] for layer in model.decoder.layers]    # [8, S_tgt, S_tgt]
    cross = [layer.cross_attn.weights[0] for layer in model.decoder.layers]          # [8, S_tgt, S_src]
    return {
        "encoder_self": torch.stack(encoder_self).cpu(),                       # [3, 8, S_src, S_src]
        "decoder_self": torch.stack(decoder_self).cpu(),                       # [3, 8, S_tgt, S_tgt]
        "cross": torch.stack(cross).cpu(),                                     # [3, 8, S_tgt, S_src]
    }


def translate_and_collect(model, number, device="cpu"):
    # 4072 -> the model's own words + all attention maps + the labels for the picture axes
    src = encode_number(number, device=device)                                 # [[7, 3, 10, 5]]
    ids = greedy_decode(model, src)                                            # [[1, 11, 27, 20, 31, 2]]
    tgt_in = ids[:, :-1]                                                       # [[1, 11, 27, 20, 31]]
    return {
        "words": ids_to_words(ids[0].tolist()),                                # "four thousand seventy two"
        "weights": collect_weights(model, src=src, tgt_in=tgt_in),
        "digits": src_tokens(number),                                          # ["4", "0", "7", "2"]
        "page": [TGT_VOCAB.itos[i] for i in tgt_in[0].tolist()],               # ["<sos>", "four", ..., "two"]
        "written": [TGT_VOCAB.itos[i] for i in ids[0, 1:].tolist()],           # ["four", ..., "two", "<eos>"]
    }


def map_labels(result, kind):
    # (row labels, column labels) for one kind of map
    row_key, col_key = Map_axes[kind]
    return result[row_key], result[col_key]


# ---------- Pictures ----------
def plot_heads(grids, row_labels, col_labels, title, path):
    # grids [layers, heads, rows, cols] -> one PNG: a grid of small heatmaps, rows = layers, cols = heads
    n_layers, n_heads = grids.shape[0], grids.shape[1]
    fig, axes = plt.subplots(n_layers, n_heads, figsize=(2.2 * n_heads, 2.4 * n_layers), squeeze=False)
    for layer in range(n_layers):
        for head in range(n_heads):
            ax = axes[layer][head]
            ax.imshow(grids[layer, head].numpy(), cmap="Blues", vmin=0, vmax=1)    # same color scale everywhere
            ax.set_xticks(range(len(col_labels)), labels=col_labels, fontsize=7, rotation=90)
            ax.set_yticks(range(len(row_labels)), labels=row_labels, fontsize=7)
            ax.set_title(f"layer {layer + 1} head {head + 1}", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)                   # make attention_maps/ if missing
    fig.savefig(path, dpi=120)
    plt.close(fig)                                                             # free the memory
    return path


def plot_average(grid, row_labels, col_labels, title, path):
    # grid [rows, cols] -> one big heatmap with the number written in every cell
    fig, ax = plt.subplots(figsize=(1.2 * len(col_labels) + 2, 0.8 * len(row_labels) + 1.5))
    image = ax.imshow(grid.numpy(), cmap="Blues", vmin=0, vmax=1)
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            value = grid[r, c].item()
            ax.text(c, r, f"{value:.2f}", ha="center", va="center",
                    color="white" if value > 0.5 else "black")                # readable on dark and light cells
    ax.set_xticks(range(len(col_labels)), labels=col_labels)
    ax.set_yticks(range(len(row_labels)), labels=row_labels)
    ax.set_title(title)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# ---------- Compare heads ----------
def summarize_heads(grids, row_labels, col_labels):
    # one line per (layer, head): for every row, the column it looks at most and how much
    lines = []
    for layer in range(grids.shape[0]):
        for head in range(grids.shape[1]):
            best_values, best_cols = grids[layer, head].max(dim=-1)           # per row: biggest weight + its column
            parts = [f"{row}->{col_labels[c]} {v:.2f}"
                     for row, c, v in zip(row_labels, best_cols.tolist(), best_values.tolist())]
            lines.append(f"layer {layer + 1} head {head + 1} | " + " | ".join(parts))
    return lines


# ---------- The one real run ----------
def report(number=4072, checkpoint_path=Checkpoint_path, out_dir=Out_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, checkpoint = load_checkpoint(path=checkpoint_path, device=device)
    result = translate_and_collect(model, number=number, device=device)
    print(f"checkpoint from epoch {checkpoint['epoch']}")
    print(f"{number} -> {result['words']}")

    paths = []
    for kind in Kinds:
        rows, cols = map_labels(result, kind=kind)
        paths.append(plot_heads(grids=result["weights"][kind], row_labels=rows, col_labels=cols,
                                title=Titles[kind], path=os.path.join(out_dir, f"{kind}_attention.png")))

    rows, cols = map_labels(result, kind="cross")
    last_layer_average = result["weights"]["cross"][-1].mean(dim=0)            # [8, 5, 4] -> [5, 4]
    paths.append(plot_average(grid=last_layer_average, row_labels=rows, col_labels=cols,
                              title="Cross-attention, layer 3, 8 heads averaged",
                              path=os.path.join(out_dir, "cross_attention_average.png")))

    print("\ncross-attention: the digit each word reads most (weight)")
    for line in summarize_heads(grids=result["weights"]["cross"], row_labels=rows, col_labels=cols):
        print(line)
    print("\nsaved:")
    for path in paths:
        print(f"  {path}")
    return paths