#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Beeswarm of per-state exposure sensitivity beta.

For each (state, representation) fits an OLS slope of log P(distractor) vs k
using k in {0,1,2,4,8}. One dot = one state. Overlays the group mean with a
95% CI whisker.

Input: the per-trial table written by code/analyze_results.py
(default ../results/analysis/per_trial.csv; override with --per_trial).

Style: serif fonts, muted palette, two-layer translucent scatter (soft halo
plus inner dot), thin whisker, no top/right spine.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

HERE = Path(__file__).resolve().parent
# Written by the analyze stage (bash run_full.sh / launch.sh); override with --per_trial.
PER_TRIAL = HERE.parent / "results" / "analysis" / "per_trial.csv"
BASE_STEM = "beeswarm_beta_logp_distractor_9b"
K_LEVELS = (0, 1, 2, 4, 8)

SEED = 42
rng = np.random.default_rng(SEED)

LABEL_SIZE  = 14
TICK_SIZE   = 12
LEGEND_SIZE = 10.5
FIGSIZE     = (5.8, 5.2)

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
RAST_C  = "#E08B2E"
STRU_C  = "#D95D79"
TEXT_C2 = "#BFD5F5"
RAST_C2 = "#F5D8B0"
STRU_C2 = "#F2C2CB"
GRID_C  = "#E5EAF2"
ZERO_C  = "#B3BAC5"
# Swarm dot opacity: halo and core.
HALO_ALPHA = 0.045
CORE_ALPHA = 0.17

REPS = [
    ("text",       TEXT_C, TEXT_C2, "Text",              "o"),
    ("rasterized", RAST_C, RAST_C2, "Rasterized",        "s"),
    ("structural", STRU_C, STRU_C2, "Structural Visual", "D"),
]

# Alternative palette (--palette house), shared with the cross-model readout
# and behavioral figures: (main color, pale halo tint) per representation.
HOUSE = {
    "text":       ("#4A6DA7", "#C9D4E6"),
    "rasterized": ("#D9A441", "#F0E3C4"),
    "structural": ("#9B8EC4", "#DFDAEC"),
}


def load_groups_from_per_trial(path: Path, metric: str = "logp_distractor_action"):
    """Read results/analysis/per_trial.csv -> {(state_id, rep): [(k, y), ...]}."""
    groups: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    valid_reps = {r[0] for r in REPS}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rep = row["representation"]
            if rep not in valid_reps:
                continue
            try:
                k = float(row["level"])
                y = float(row[metric])
            except (ValueError, KeyError):
                continue
            if k < 0 or not np.isfinite(y):
                continue
            groups[(row["state_id"], rep)].append((k, y))
    return groups


def load_per_state_slopes(metric: str = "logp_distractor_action", per_trial: Path = PER_TRIAL):
    """OLS-fit log P(distractor) vs k for each (state, representation); return {rep: np.ndarray}."""
    if not per_trial.is_file():
        raise SystemExit(f"per-trial table not found: {per_trial}\n"
                         "Run the rasterized-interference analysis first (run_full.sh).")
    groups = load_groups_from_per_trial(per_trial, metric)

    per_state: dict[str, list[float]] = {r[0]: [] for r in REPS}
    for (_sid, rep), pts in groups.items():
        if len(pts) < 2:
            continue
        xs = np.array([p[0] for p in pts])
        ys = np.array([p[1] for p in pts])
        if xs.std() == 0:
            continue
        slope, _ = np.polyfit(xs, ys, 1)
        per_state[rep].append(float(slope))
    return {rep: np.asarray(vs, dtype=float) for rep, vs in per_state.items()}


def beeswarm_x(y: np.ndarray, center: float, width: float = 0.22, bin_frac: float = 0.025) -> np.ndarray:
    """Deterministic beeswarm layout: bin y, then symmetric-spread duplicates within each bin."""
    if len(y) == 0:
        return np.array([])
    y_min, y_max = float(y.min()), float(y.max())
    span = max(y_max - y_min, 1e-9)
    bin_w = span * bin_frac
    order = np.argsort(y)
    positions = np.zeros_like(y, dtype=float)
    bucket_counts: dict[int, int] = {}
    for idx in order:
        b = int((y[idx] - y_min) / bin_w) if bin_w > 0 else 0
        c = bucket_counts.get(b, 0)
        offset = ((c + 1) // 2) * (1 if c % 2 == 0 else -1)
        positions[idx] = offset
        bucket_counts[b] = c + 1
    max_off = max(1, np.max(np.abs(positions)))
    return center + (positions / max_off) * width


def save(fig, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # No bbox_inches="tight": tight cropping resizes to content, so figures with
    # different tick-label widths / data ranges come out at different pixel sizes.
    # Fixed figsize + fixed subplots_adjust margins guarantee an identical canvas
    # and an identical axes box (same x/y axis lengths) across runs.
    fig.savefig(out_dir / f"{stem}.png", dpi=600)
    fig.savefig(out_dir / f"{stem}.pdf")
    plt.close(fig)


def main() -> None:
    global REPS
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", type=str, default=str(HERE))
    ap.add_argument("--per_trial", nargs="?", const=str(PER_TRIAL), default=str(PER_TRIAL),
                    help="analyze-stage per_trial.csv (default and bare flag: "
                         "../results/analysis/per_trial.csv)")
    ap.add_argument("--palette", choices=("original", "house"),
                    default="original",
                    help="house = the cross-model figure palette, written to "
                         "a separate output stem")
    ap.add_argument("--stem", default=None,
                    help="output stem; default: BASE_STEM for original, "
                         "BASE_STEM + '_house' for house")
    args = ap.parse_args()
    if args.palette == "house":
        REPS = [(rep, HOUSE[rep][0], HOUSE[rep][1], lab, mk)
                for rep, _c, _l, lab, mk in REPS]
    stem = args.stem or (BASE_STEM if args.palette == "original"
                         else BASE_STEM + "_house")

    per_state = load_per_state_slopes(
        metric="logp_distractor_action",
        per_trial=Path(args.per_trial))

    print("Per-representation β summary (slope of log P(distractor) vs k):")
    for rep, color, _light, label, _marker in REPS:
        y = per_state[rep]
        m = float(np.mean(y))
        sem = float(np.std(y, ddof=1) / np.sqrt(len(y)))
        print(f"  {label:20s}  n={len(y):3d}  mean={m:+.4f}  95%CI=±{1.96*sem:.4f}")

    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.subplots_adjust(left=0.18, right=0.95, bottom=0.14, top=0.97)

    xpos = [1, 2, 3]
    means = []
    for xi, (rep, color, light, _label, marker) in zip(xpos, REPS):
        y = per_state[rep]
        if len(y) == 0:
            continue
        x = beeswarm_x(y, center=xi, width=0.24)

        # Airy, translucent swarm: a very soft wide halo plus a faint small
        # core so the cloud reads as diffuse rather than a solid mass.
        ax.scatter(x, y, s=44, color=light, alpha=HALO_ALPHA,
                   linewidths=0, zorder=1, rasterized=True)
        ax.scatter(x, y, s=7, color=color, alpha=CORE_ALPHA,
                   linewidths=0, zorder=2, rasterized=True)

        m = float(np.mean(y))
        sem = float(np.std(y, ddof=1) / np.sqrt(len(y)))
        ci95 = 1.96 * sem
        means.append((xi, m, ci95, color, marker))

    for xi, m, ci95, color, marker in means:
        ax.plot([xi, xi], [m - ci95, m + ci95],
                color=color, lw=1.4, solid_capstyle="round", zorder=4)
        cap = 0.06
        ax.plot([xi - cap, xi + cap], [m - ci95, m - ci95],
                color=color, lw=1.4, solid_capstyle="round", zorder=4)
        ax.plot([xi - cap, xi + cap], [m + ci95, m + ci95],
                color=color, lw=1.4, solid_capstyle="round", zorder=4)
        ax.scatter([xi], [m], s=36, marker=marker,
                   facecolor=color, edgecolor=color,
                   linewidth=0.45, zorder=5)

    ax.axhline(0.0, ls=(0, (4, 3)), lw=0.85, color=ZERO_C, zorder=0)

    ax.set_xlim(0.5, 3.5)
    all_vals = np.concatenate([per_state[r[0]] for r in REPS])
    y_lo = min(0.0, float(all_vals.min())) - 0.06
    y_hi = float(all_vals.max()) + 0.08
    ax.set_ylim(y_lo, y_hi)

    ax.set_xticks(xpos)
    ax.set_xticklabels([r[3] for r in REPS])
    ax.yaxis.grid(True, color=GRID_C, lw=0.9)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    ax.set_ylabel("Distractor effect per exposure")
    ax.tick_params(axis="x", width=1.0, length=4.5)
    ax.tick_params(axis="y", width=1.0, length=4.5)

    handles = [
        Line2D([0], [0], linestyle="None", marker=m, ms=5.4,
               mfc=c, mec=c, mew=0.45, label=lab)
        for (_, c, _l, lab, m) in REPS
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False,
              handletextpad=0.45, borderaxespad=0.35, labelspacing=0.28)

    out_dir = Path(args.output_dir)
    save(fig, out_dir, stem)
    print(f"wrote {out_dir / (stem + '.png')}")
    print(f"wrote {out_dir / (stem + '.pdf')}")


if __name__ == "__main__":
    main()
