#!/usr/bin/env python3
"""Draw confusion_matrices/summary.png, the results figure at the top of the README.

    python plot_summary.py

Reads the result tables the notebooks write to results/ (sklearn_pipeline.ipynb, llama_few_shot.ipynb); the RNN
score is taken from rnn.ipynb's test output.
"""
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from config import CFG

RNN_TEST_MACRO_F1 = 0.509      # rnn.ipynb, BiGRU on the test set
ALWAYS_MEDIUM = 0.225          # macro-F1 of always answering "Medium" on the test set

SURF, INK, MUTED, GRAY, BLUE, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#b4b2a9", "#2a78d6", "#ecebe6"
SKLEARN_NAMES = {"perceptron": "Perceptron", "lsvm": "Linear SVM", "svm": "RBF SVM",
                 "mlp_c": "MLP classifier", "mlp_r": "MLP regressor"}


def load():
    res = CFG.results_dir
    sk = pd.read_csv(res / "sklearn_tuned_results.csv", index_col="model")["test_macro_f1"]
    llama = pd.read_csv(res / "llama3.1-8b_results.csv", index_col="variant")["test_macro_f1"]
    new = pd.read_csv(res / "llama3.1-8b_new_problems.csv", index_col="model")

    bars = [("Always Medium", ALWAYS_MEDIUM, False),
            ("Llama 3.1 8B zero-shot", llama["0-shot calibrated"], True)]
    bars += [(SKLEARN_NAMES[m], v, False) for m, v in sk.items()]
    bars += [("BiGRU (RNN)", RNN_TEST_MACRO_F1, False),
             ("Llama 3.1 8B 6-shot", llama["6-shot raw"], True),
             ("Llama 6-shot + Easy/Hard shift", new.loc["Llama 6-shot F1-tuned", "old_test_macro_f1"], True)]
    bars.sort(key=lambda b: b[1])

    rows = {"TF-IDF (no memory)": "TF-IDF + logistic regression (no memory)",
            "Llama 6-shot, full statement": "Llama 6-shot calibrated",
            "Llama 6-shot + shift": "Llama 6-shot F1-tuned",
            "Llama 6-shot, title only": "Llama 6-shot calibrated, title only"}
    pairs = [(label, new.loc[key, "old_test_macro_f1"], new.loc[key, "new_macro_f1"]) for label, key in rows.items()]
    text_only = [v for _, v, llm in bars if not llm and v > ALWAYS_MEDIUM]
    return bars, pairs, (min(text_only), max(text_only))


def style(ax):
    ax.set_facecolor(SURF)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, lw=1)
    ax.set_axisbelow(True)


def draw(bars, pairs, band, out: Path):
    mpl.rcParams.update({"font.size": 10, "axes.edgecolor": "#c9c8c1", "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": INK})
    fig, (a, b) = plt.subplots(1, 2, figsize=(13.5, 5.2), gridspec_kw={"width_ratios": [1.2, 1]}, facecolor=SURF)
    style(a), style(b)

    # left: test macro-F1 of every model
    a.axvspan(*band, color="#eef4fc", zorder=0)
    a.text(sum(band) / 2, len(bars) - .35, "text-only models", ha="center", va="bottom", color=MUTED, fontsize=8.5)
    a.barh(range(len(bars)), [v for _, v, _ in bars], height=.62,
           color=[BLUE if llm else GRAY for _, _, llm in bars], zorder=2)
    for i, (_, v, _) in enumerate(bars):
        a.text(v + .008, i, f"{v:.2f}", va="center", color=INK, fontsize=9)
    a.set_yticks(range(len(bars)), [name for name, _, _ in bars])
    a.set_xlim(0, .7)
    a.set_ylim(-.6, len(bars) + .2)
    a.set_xlabel("macro-F1 on the 474-problem test set (±0.05 is noise)")
    a.set_title("Text-only models land near 0.50; Llama is clearly above", loc="left", color=INK, fontsize=11, pad=10)

    # right: old test set vs problems published after Llama's training
    for i, (_, old, new) in enumerate(pairs):
        b.plot([old, new], [i, i], color="#d6d5ce", lw=2, zorder=1, solid_capstyle="round")
        b.scatter([old], [i], s=70, color=GRAY, zorder=2, edgecolor=SURF, linewidth=2)
        b.scatter([new], [i], s=70, color=BLUE, zorder=3, edgecolor=SURF, linewidth=2)
        d = new - old
        b.text(max(old, new) + .012, i, f"{d:+.2f}", va="center", color=INK if abs(d) > .03 else MUTED, fontsize=9)
    b.set_yticks(range(len(pairs)), [label for label, _, _ in pairs])
    b.set_xlim(.42, .68)
    b.set_ylim(len(pairs) - .4, -.9)
    b.scatter([], [], s=70, color=GRAY, label="old test set (Llama may remember these)")
    b.scatter([], [], s=70, color=BLUE, label="876 problems published after Llama's training")
    b.legend(loc="upper center", bbox_to_anchor=(.4, -.14), frameon=False, fontsize=8.5, labelcolor=MUTED)
    b.set_title("Memory check: only title-only gets worse", loc="left", color=INK, fontsize=11, pad=10)
    b.set_xlabel("macro-F1")

    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor=SURF)
    print(f"saved {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=CFG.figures_dir / "summary.png")
    draw(*load(), ap.parse_args().out)
