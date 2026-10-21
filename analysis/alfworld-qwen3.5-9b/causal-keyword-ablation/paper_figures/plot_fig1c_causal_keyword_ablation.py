#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Figure 1c: mean P(distractor) at k = 8 across three procedures
(original / target replaced / control replaced), Text vs Visual, with the
mean value printed above every bar.

Metric: per-token mean action probability (p_distractor_mean).  The summed
metric collapses towards zero for near-deterministic generations; the
length-normalised metric is comparable across models with different output
sharpness.  Input: the per-trial table written by code/analyze_results.py
(default ../results/analysis/per_trial.csv; override with --per_trial).

    python plot_fig1c_causal_keyword_ablation.py [--output_dir <dir>] [--backbone 9b] [--per_trial <csv>]
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SEED = 42
rng = np.random.default_rng(SEED)

LABEL_SIZE  = 13
TICK_SIZE   = 11
LEGEND_SIZE = 10.5
BARLAB_SIZE = 9.5
FIGSIZE     = (4.15, 3.85)

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

TEXT_C  = "#2F73CF"
VIS_C   = "#D95D79"
TEXT_C2 = "#BFD5F5"
VIS_C2  = "#F2C2CB"
GRID_C  = "#E5EAF2"

CONDS = [
    ("original",        "Original"),
    ("counterfactual",  "Target\nreplaced"),
    ("matched_control", "Control\nreplaced"),
]

REPS = [
    ("text",       TEXT_C, TEXT_C2, "Text",   "o"),
    ("structural", VIS_C,  VIS_C2,  "Visual", "D"),
]

HERE = Path(__file__).resolve().parent
DEFAULT_PER_TRIAL = HERE.parent / "results" / "analysis" / "per_trial.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_trial", type=str, default=str(DEFAULT_PER_TRIAL),
                    help="per_trial.csv written by code/analyze_results.py")
    ap.add_argument("--output_dir", type=str, default=str(HERE))
    ap.add_argument("--backbone", type=str, default="",
                    help="suffix appended to output filenames, e.g. 9b")
    args = ap.parse_args()

    # results/analysis/per_trial.csv written by code/analyze_results.py
    per_trial = Path(args.per_trial)
    if not per_trial.is_file():
        raise SystemExit(f"per-trial table not found: {per_trial}\n"
                         "Run the causal-keyword-ablation analysis first (run_full.sh).")
    df = pd.read_csv(per_trial)
    df = df[df.level == 8]
    df = df[df["representation"].isin([r[0] for r in REPS])]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.18, top=0.97)

    x = np.arange(len(CONDS), dtype=float)
    width = 0.22
    offsets = {"text": -width / 2 - 0.01, "structural": +width / 2 + 0.01}

    ymax_data = 0.0
    for rep, _, _, _, _ in REPS:
        for cond, _ in CONDS:
            arr = df[(df["representation"] == rep) & (df["version"] == cond)][
                "p_distractor_mean"
            ].to_numpy()
            if arr.size:
                ymax_data = max(ymax_data, arr.mean())
    ymax = min(1.0, ymax_data * 1.35)
    ax.set_ylim(0, ymax)

    labels = []  # (x_pos, value, color)
    for rep, color, light, _label, _marker in REPS:
        means = []
        for cond, _ in CONDS:
            arr = df[(df["representation"] == rep) & (df["version"] == cond)][
                "p_distractor_mean"
            ].to_numpy()
            means.append(float(np.nanmean(arr)) if arr.size else np.nan)

        xs = x + offsets[rep]
        ax.bar(
            xs, means, width=width,
            color=light, edgecolor=color,
            linewidth=1.0, alpha=1.0, zorder=2,
        )
        # nudge Visual (pink) labels right so they clear the Text bar edge
        lab_dx = 0.07 if rep == "structural" else 0.0
        for xi, m in zip(xs, means):
            labels.append((xi + lab_dx, m, color))

    # numeric value on top of every bar
    pad = 0.015 * ymax
    for xi, m, color in labels:
        if np.isnan(m):
            continue
        ax.text(
            xi, m + pad, f"{m:.2f}",
            ha="center", va="bottom",
            fontsize=BARLAB_SIZE, color=color, zorder=5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in CONDS])
    ax.set_ylabel("Distractor action probability")
    ax.set_xlim(-0.55, len(CONDS) - 0.45)
    ax.yaxis.grid(True, color=GRID_C, lw=0.9)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    ax.tick_params(width=1.0, length=4.5)

    handles = [
        Line2D([0], [0], linestyle="None", marker="o", ms=5.5,
               mfc=TEXT_C, mec=TEXT_C, mew=0.4, label="Text"),
        Line2D([0], [0], linestyle="None", marker="D", ms=5.2,
               mfc=VIS_C,  mec=VIS_C,  mew=0.4, label="Visual"),
    ]
    ax.legend(
        handles=handles,
        loc="upper right",
        frameon=False,
        handletextpad=0.4,
        borderaxespad=0.35,
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = "figure2_bars_p_distractor_v4_mean" + (f"_{args.backbone}" if args.backbone else "")
    fig.savefig(out / f"{stem}.png",
                dpi=600, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out / f"{stem}.pdf",
                bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("Saved to:", out / f"{stem}.png")


if __name__ == "__main__":
    main()
