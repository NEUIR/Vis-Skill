#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Figure 3c: executable action selection, both backbones in one axes.

Four lines, (Qwen3.5-9B / Gemma-4-31B-it) x (Text / Visual Skill): the rate at
which the competing (distractor) action is selected at each exposure level k.
Shaded bands are Wilson 95% intervals over the decision states of each cell.
Only trials in which every candidate action was scored are used.

Inputs are the per-trial tables written by code/analyze_results.py of the two
probability-to-behavior runs:
  Qwen3.5-9B    : ../results/analysis/per_trial.csv
  Gemma-4-31B-it: ../../../alfworld-gemma-4-31b/probability-to-behavior/results/analysis/per_trial.csv

    python plot_fig3c_behavioral_realization.py [--qwen_csv <csv>] [--gemma_csv <csv>] [--output_dir <dir>]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

LABEL_SIZE = 14
TICK_SIZE = 12
LEGEND_SIZE = 10.5
ENDLAB_SIZE = 8.5
FIGSIZE = (5.8, 5.2)          # same canvas as plot_fig3b_symbol_readout
# Same margins as plot_fig3b_symbol_readout so the axes box matches.
MARGINS = dict(left=0.18, right=0.95, bottom=0.14, top=0.97)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.labelsize": LABEL_SIZE,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
    "legend.fontsize": LEGEND_SIZE,
    "axes.linewidth": 1.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

GRID_C = "#E5EAF2"

# Cool tones for Qwen, warm tones for Gemma (same palette as Fig. 3b).
QWEN_TEXT_C = "#4A6DA7"
QWEN_VIS_C = "#9B8EC4"
GEMMA_TEXT_C = "#D9A441"
GEMMA_VIS_C = "#B06A5B"
BAND_ALPHA = 0.07             # light washes, since four bands overlap

K_LEVELS = [0, 1, 2, 4, 8]

HERE = Path(__file__).resolve().parent
DEFAULT_QWEN_CSV = HERE.parent / "results" / "analysis" / "per_trial.csv"
DEFAULT_GEMMA_CSV = (HERE.parents[2] / "alfworld-gemma-4-31b" / "probability-to-behavior"
                     / "results" / "analysis" / "per_trial.csv")

# (label, backbone, representation, colour, marker, end-label dy in points)
SERIES = [
    ("Qwen3.5-9B · Text",   "qwen",  "text",       QWEN_TEXT_C,  "o", 0.0),
    ("Qwen3.5-9B · Visual", "qwen",  "structural", QWEN_VIS_C,   "D", 7.0),
    ("Gemma-4-31B · Text",  "gemma", "text",       GEMMA_TEXT_C, "o", 0.0),
    ("Gemma-4-31B · Visual", "gemma", "structural", GEMMA_VIS_C, "D", -7.0),
]


def wilson(k, n, z=1.959963985):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def load(csv_path: Path) -> pd.DataFrame:
    if not csv_path.is_file():
        raise SystemExit(f"per-trial table not found: {csv_path}\n"
                         "Run the probability-to-behavior analysis first (run_full.sh).")
    df = pd.read_csv(csv_path)
    df = df[df["n_scored"] == df["n_candidates"]].copy()
    df = df[df.representation.isin(["text", "structural"])
            & df.level.isin(K_LEVELS)].copy()
    df["chose_distractor"] = df["chose_distractor"].astype(float)
    return df


def series(df: pd.DataFrame, rep: str):
    sub = df[df.representation == rep]
    g = sub.groupby("level")["chose_distractor"]
    k_sel = g.sum().reindex(K_LEVELS).to_numpy(float)
    n = g.size().reindex(K_LEVELS).to_numpy(int)
    mean = k_sel / n
    band = np.array([wilson(k, nn) for k, nn in zip(k_sel, n)])
    return mean, band[:, 0], band[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen_csv", type=str, default=str(DEFAULT_QWEN_CSV),
                    help="per_trial.csv of the Qwen3.5-9B probability-to-behavior run")
    ap.add_argument("--gemma_csv", type=str, default=str(DEFAULT_GEMMA_CSV),
                    help="per_trial.csv of the Gemma-4-31B-it probability-to-behavior run")
    ap.add_argument("--output_dir", type=str, default=str(HERE))
    args = ap.parse_args()

    frames = {"qwen": load(Path(args.qwen_csv)), "gemma": load(Path(args.gemma_csv))}

    top = 0.0
    drawn = []
    for label, backbone, rep, color, marker, dy in SERIES:
        mean, lo, hi = series(frames[backbone], rep)
        drawn.append((label, mean, lo, hi, color, marker, dy))
        top = max(top, float(np.nanmax(hi)))
    top_lim = top * 1.18

    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.subplots_adjust(**MARGINS)

    for label, mean, lo, hi, color, marker, dy in drawn:
        ax.fill_between(K_LEVELS, lo, hi, color=color, alpha=BAND_ALPHA,
                        lw=0, zorder=2)
        ax.plot(K_LEVELS, mean, color=color, lw=1.8, solid_capstyle="round",
                zorder=3)
        ax.scatter(K_LEVELS, mean, s=32, marker=marker, facecolor=color,
                   edgecolor=color, linewidth=1.0, zorder=4)
        ax.annotate(f"{100 * mean[-1]:.1f}", (K_LEVELS[-1], mean[-1]),
                    xytext=(6, dy), textcoords="offset points",
                    ha="left", va="center", fontsize=ENDLAB_SIZE, color=color,
                    zorder=5)

    ax.set_xticks(K_LEVELS)
    ax.set_xlim(-0.4, 9.4)          # room for the k=8 endpoint labels
    ax.set_ylim(0.0, top_lim)
    ax.set_yticks(np.arange(0.0, top_lim, 0.04))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:.0f}"))
    ax.set_xlabel(r"Exposure level $k$")
    ax.set_ylabel("Competing-action selection (%)")
    ax.yaxis.grid(True, color=GRID_C, lw=0.9)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    ax.tick_params(width=1.0, length=4.5)

    ax.legend([Line2D([0], [0], color=c, lw=1.8, marker=m, ms=5.4,
                      mfc=c, mec=c, mew=1.0)
               for *_, c, m, _ in drawn],
              [lb for lb, *_ in drawn],
              loc="upper left", frameon=False, handlelength=2.2,
              handletextpad=0.5, labelspacing=0.3, borderaxespad=0.4)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / "figure_behavioral_realization_crossmodel"
    # fixed canvas (no bbox_inches="tight") so the size matches Fig. 3b
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)

    for label, mean, lo, hi, _c, _m, _dy in drawn:
        print(f"  {label:22s} " + "  ".join(
            f"k{k}={100 * m:.2f}%[{100 * l:.2f},{100 * h:.2f}]"
            for k, m, l, h in zip(K_LEVELS, mean, lo, hi)))
    print(f"Saved: {stem}.png / .pdf")


if __name__ == "__main__":
    main()
