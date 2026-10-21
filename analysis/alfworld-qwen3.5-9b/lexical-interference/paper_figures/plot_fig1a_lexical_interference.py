#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Figure 1a: distractor-action probability versus keyword exposure k.

Metric: per-token mean action probability (p_distractor_mean).  Every
admissible candidate action is scored by forced decoding, its token
log-probabilities are averaged per token (length-normalised), and the scores
are softmaxed over the candidate set.  The length-normalised metric is used
because the summed metric collapses towards zero for near-deterministic
generations, which would hide the per-trial spread on a linear axis.

Jittered points are individual trials; lines are means with +/- s.e.m.
ribbons.  y-limits are data-driven (upper edge = 1.05 * max trial, rounded
up to 0.05, capped at 0.5).

Input: the per-trial table written by code/analyze_results.py
(default ../results/analysis/per_trial.csv; override with --per_trial).
Only trials in which every candidate action was scored are used.

    python plot_fig1a_lexical_interference.py --output_dir <dir> [--backbone 9b] [--per_trial <csv>]
"""

from pathlib import Path
import argparse
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

METRIC   = "p_distractor_mean"
K_LEVELS = [0, 1, 2, 4, 8]
REPS = [
    ("text",   TEXT_C, TEXT_C2, "Text",   "o"),
    ("visual", VIS_C,  VIS_C2,  "Visual", "D"),
]

HERE = Path(__file__).resolve().parent
DEFAULT_PER_TRIAL = HERE.parent / "results" / "analysis" / "per_trial.csv"


def load(csv_path: Path) -> pd.DataFrame:
    # results/analysis/per_trial.csv written by code/analyze_results.py
    if not csv_path.is_file():
        raise SystemExit(f"per-trial table not found: {csv_path}\n"
                         "Run the lexical-interference analysis first (run_full.sh).")
    df = pd.read_csv(csv_path)
    df = df[df["n_scored"] == df["n_candidates"]].copy()
    df = df[
        df["representation"].isin(["text", "visual"])
        & df["level"].isin(K_LEVELS)
    ].copy()
    return df


def save(fig, out_dir: Path, stem: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", dpi=600, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def panel_a(df: pd.DataFrame, out_dir: Path, backbone: str = ""):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.18, top=0.97)

    for rep, color, light, label, marker in REPS:
        sub = df[df.representation == rep]
        grp = sub.groupby("level")[METRIC]
        means = grp.mean().reindex(K_LEVELS).to_numpy()
        sems  = grp.sem().reindex(K_LEVELS).to_numpy()

        for k in K_LEVELS:
            y = sub.loc[sub.level == k, METRIC].to_numpy()
            x = k + rng.normal(0, 0.055, size=y.size)
            ax.scatter(x, y, s=38, color=light, alpha=0.05,
                       linewidths=0, zorder=1, rasterized=True)
            ax.scatter(x, y, s=10, color=color, alpha=0.12,
                       linewidths=0, zorder=2, rasterized=True)

        ax.fill_between(K_LEVELS, means - sems, means + sems,
                         color=color, alpha=0.10, lw=0, zorder=3)
        ax.plot(K_LEVELS, means, color=color, lw=1.45,
                solid_capstyle="round", zorder=4)
        ax.scatter(K_LEVELS, means, s=30, marker=marker,
                   facecolor=color, edgecolor=color,
                   linewidth=0.4, zorder=5)

    handles = [
        Line2D([0], [0], color=TEXT_C, lw=1.45, marker="o", ms=5.5,
               mfc=TEXT_C, mec=TEXT_C, mew=0.4, label="Text"),
        Line2D([0], [0], color=VIS_C, lw=1.45, marker="D", ms=5.2,
               mfc=VIS_C, mec=VIS_C, mew=0.4, label="Visual"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False,
              handlelength=1.8, handletextpad=0.5)

    ax.set_xlim(-0.28, 8.65)
    # data-driven y-range: upper edge = 1.05 * max trial rounded up to 0.05,
    # capped at 0.5; trials above the cap are omitted from the scatter only
    upper = min(np.ceil(df[METRIC].max() * 1.05 / 0.05) * 0.05, 0.5)
    step = 0.05 if upper <= 0.35 else (0.1 if upper <= 0.6 else 0.2)
    ax.set_ylim(-0.03 * upper, upper)
    ax.set_xticks(K_LEVELS)
    ax.set_yticks(np.arange(0, upper + 1e-9, step))
    ax.yaxis.grid(True, color=GRID_C, lw=0.9)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    ax.set_xlabel(r"Keyword occurrences $k$")
    ax.set_ylabel("Distractor action probability")
    ax.tick_params(width=1.0, length=4.5)

    stem = "panel_a_pdistractor_vs_k" + (f"_{backbone}" if backbone else "")
    save(fig, out_dir, stem)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_trial", type=str, default=str(DEFAULT_PER_TRIAL),
                    help="per_trial.csv written by code/analyze_results.py")
    ap.add_argument("--output_dir", type=str, default=str(HERE))
    ap.add_argument("--backbone", type=str, default="",
                    help="suffix appended to output filenames, e.g. 9b")
    args = ap.parse_args()

    df = load(Path(args.per_trial))
    out_dir = Path(args.output_dir)

    panel_a(df, out_dir, args.backbone)

    print(f"Saved figures to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
