#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Lexical-interference figures from the summed-logP metric (p_distractor_sum).

Panel A: distractor probability versus keyword exposure k (trials + mean +/- s.e.m.).
Panel B: per-state change in distractor probability, k=8 minus k=0 (Figure 1b).
Panel C: distractor rank among candidates at k=0 versus k=8.

Input: the per-trial table written by code/analyze_results.py
(default ../results/analysis/per_trial.csv; override with --per_trial).
Only trials in which every candidate action was scored are used. All panels
draw their jitter from one shared RNG in the order A, B, C, so Figure 1b is
the panel B written by the default `--panels abc` run.

    python plot_fig1b_delta_p_distribution.py --output_dir <dir> [--backbone 9b] [--panels abc] [--per_trial <csv>]
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
from scipy.ndimage import gaussian_filter1d

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
ZERO_C  = "#B3BAC5"
GRAY_C  = "#9AA1AB"
GRAY_2  = "#DDE2E8"

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


def pivot08(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    sub = df[df.level.isin([0, 8])]
    return sub.pivot_table(
        index="state_id",
        columns=["representation", "level"],
        values=metric,
    )


def discrete_rank_density(data, xs, sigma=0.95):
    data = np.asarray(data, dtype=float)
    ranks = np.clip(np.rint(data).astype(int), 1, 16)
    counts = np.bincount(ranks, minlength=17)[1:17].astype(float)
    smooth = gaussian_filter1d(counts, sigma=sigma, mode="reflect")
    if smooth.max() > 0:
        smooth /= smooth.max()
    rank_grid = np.arange(1, 17, dtype=float)
    return np.interp(xs, rank_grid, smooth)


# =========================================================
# Panel A
# =========================================================
def panel_a(df: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.18, top=0.97)

    for rep, color, light, label, marker in REPS:
        sub = df[df.representation == rep]
        grp = sub.groupby("level")["p_distractor_sum"]
        means = grp.mean().reindex(K_LEVELS).to_numpy()
        sems  = grp.sem().reindex(K_LEVELS).to_numpy()

        for k in K_LEVELS:
            y = sub.loc[sub.level == k, "p_distractor_sum"].to_numpy()
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
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks(K_LEVELS)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.yaxis.grid(True, color=GRID_C, lw=0.9)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    ax.set_xlabel(r"Skill word frequency $k$")
    ax.set_ylabel(r"$P(\mathrm{distractor})$")
    ax.tick_params(width=1.0, length=4.5)

    save(fig, out_dir, "panel_a_pdistractor_vs_k")


# =========================================================
# Panel B
# =========================================================
def panel_b(df: pd.DataFrame, out_dir: Path, backbone: str = ""):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.18, top=0.97)

    piv = pivot08(df, "p_distractor_sum")
    dt = (piv[("text", 8)] - piv[("text", 0)]).dropna()
    dv = (piv[("visual", 8)] - piv[("visual", 0)]).dropna()

    data = [dt.to_numpy(), dv.to_numpy()]
    floor = min(0.0, np.concatenate(data).min())

    parts = ax.violinplot(
        data, positions=[1, 2], widths=0.56,
        showmeans=False, showmedians=False, showextrema=False,
        bw_method=0.25,
    )
    for body, color in zip(parts["bodies"], [TEXT_C, VIS_C]):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.12)
        body.set_linewidth(0.85)
        verts = body.get_paths()[0].vertices
        verts[verts[:, 1] < floor, 1] = floor

    cfg = [
        (1, dt.to_numpy(), TEXT_C, TEXT_C2, "o"),
        (2, dv.to_numpy(), VIS_C,  VIS_C2,  "D"),
    ]
    for xpos, arr, color, light, marker in cfg:
        xj = xpos + rng.normal(0, 0.050, size=arr.size)
        ax.scatter(xj, arr, s=26, color=light, alpha=0.060,
                   linewidths=0, zorder=1, rasterized=True)
        ax.scatter(xj, arr, s=10, color=color, alpha=0.21,
                   linewidths=0, zorder=2, rasterized=True)

        m = arr.mean()
        ax.scatter([xpos], [m], s=34, marker=marker,
                   facecolor=color, edgecolor=color,
                   linewidth=0.45, zorder=4)

    handles = [
        Line2D([0], [0], linestyle="None", marker="o", ms=5.4,
               mfc=TEXT_C, mec=TEXT_C, mew=0.45, label="Text"),
        Line2D([0], [0], linestyle="None", marker="D", ms=5.2,
               mfc=VIS_C, mec=VIS_C, mew=0.45, label="Visual"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False,
              handletextpad=0.4, borderaxespad=0.35)

    ax.axhline(0, ls=(0, (4, 3)), lw=0.85, color=ZERO_C, zorder=0)

    ax.set_xlim(0.45, 2.55)
    ax.set_ylim(floor - 0.03, max(dt.max(), dv.max()) + 0.08)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Text", "Visual"])
    ax.yaxis.grid(True, color=GRID_C, lw=0.75)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    ax.set_ylabel(r"$\Delta P(\mathrm{distractor})$")
    ax.tick_params(width=0.95, length=4.2)

    stem = "panel_b_delta_p_distribution" + (f"_{backbone}" if backbone else "")
    save(fig, out_dir, stem)


# =========================================================
# Panel C
# =========================================================
def panel_c(df: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.subplots_adjust(left=0.20, right=0.98, bottom=0.18, top=0.97)

    piv = pivot08(df, "rank_distractor_sum")

    rows = {
        "text": 0.18,
        "visual": -0.04,
    }
    cfg = {
        "text":   dict(color=TEXT_C, light=TEXT_C2, marker="o"),
        "visual": dict(color=VIS_C,  light=VIS_C2,  marker="D"),
    }

    xs = np.linspace(0.8, 16.5, 700)

    for rep in ["text", "visual"]:
        c = cfg[rep]
        y0 = rows[rep]

        pair = piv[[(rep, 0), (rep, 8)]].dropna()
        r0 = pair[(rep, 0)].to_numpy(float)
        r8 = pair[(rep, 8)].to_numpy(float)
        m0 = r0.mean()
        m8 = r8.mean()

        d0 = discrete_rank_density(r0, xs, sigma=1.00)
        d8 = discrete_rank_density(r8, xs, sigma=0.90)

        h0 = 0.095 * d0
        h8 = 0.095 * d8

        ax.fill_between(xs, y0, y0 + h0, color=GRAY_2, alpha=0.58, lw=0, zorder=1)
        ax.plot(xs, y0 + h0, color=GRAY_C, lw=0.9, alpha=0.85, zorder=2)

        ax.fill_between(xs, y0, y0 - h8, color=c["light"], alpha=0.42, lw=0, zorder=1)
        ax.plot(xs, y0 - h8, color=c["color"], lw=1.15, alpha=0.98, zorder=2)

        ax.plot([xs.min(), xs.max()], [y0, y0],
                color=c["color"], lw=0.85, alpha=0.90, zorder=2)

        x0 = r0 + rng.normal(0, 0.035, len(r0))
        x8 = r8 + rng.normal(0, 0.035, len(r8))
        y0pts = y0 + rng.uniform(0.018, 0.060, len(r0))
        y8pts = y0 - rng.uniform(0.018, 0.060, len(r8))

        ax.scatter(x0, y0pts, s=9, color=GRAY_C, alpha=0.14,
                   linewidths=0, rasterized=True, zorder=3)
        ax.scatter(x8, y8pts, s=9, color=c["color"], alpha=0.18,
                   linewidths=0, rasterized=True, zorder=3)

        y_init = y0 + 0.070
        y_rep  = y0 - 0.070

        ax.scatter([m0], [y_init], s=34, marker="D",
                   facecolor=GRAY_C, edgecolor=GRAY_C,
                   linewidth=0.4, zorder=5)
        ax.scatter([m8], [y_rep], s=36, marker=c["marker"],
                   facecolor=c["color"], edgecolor=c["color"],
                   linewidth=0.4, zorder=5)

        if rep == "text":
            ax.annotate(
                "",
                xy=(m8 + 0.18, y_rep + 0.015),
                xytext=(m0 - 0.12, y_init - 0.012),
                arrowprops=dict(
                    arrowstyle="-|>", color=c["color"],
                    lw=1.45, mutation_scale=13,
                    shrinkA=0, shrinkB=0,
                    connectionstyle="arc3,rad=0.12",
                ),
                zorder=4,
            )
            ax.text(m0 + 1.15, y_init + 0.05,
                    f"{m0:.1f} → {m8:.1f}",
                    color=c["color"], fontsize=10.8,
                    ha="left", va="center", zorder=6)
        else:
            ax.annotate(
                "",
                xy=(m8 + 0.15, y_rep + 0.010),
                xytext=(m0 - 0.12, y_init - 0.010),
                arrowprops=dict(
                    arrowstyle="-|>", color=c["color"],
                    lw=1.45, mutation_scale=13,
                    shrinkA=0, shrinkB=0,
                    connectionstyle="arc3,rad=0.10",
                ),
                zorder=4,
            )
            ax.text(m0 + 0.95, y_init + 0.04,
                    f"{m0:.1f} → {m8:.1f}",
                    color=c["color"], fontsize=10.8,
                    ha="left", va="center", zorder=6)

    handles = [
        Line2D([0], [0], linestyle="None", marker="D", ms=5.6,
               mfc=GRAY_C, mec=GRAY_C, mew=0.4, label=r"Initial ($k=0$)"),
        Line2D([0], [0], linestyle="None", marker="o", ms=5.8,
               mfc=TEXT_C, mec=TEXT_C, mew=0.4, label=r"Repeated Text ($k=8$)"),
        Line2D([0], [0], linestyle="None", marker="D", ms=5.4,
               mfc=VIS_C, mec=VIS_C, mew=0.4, label=r"Repeated Visual ($k=8$)"),
    ]
    ax.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        frameon=False,
        handletextpad=0.45,
        labelspacing=0.28,
        borderaxespad=0.1,
    )

    ax.set_xlim(0.6, 16.5)
    ax.set_ylim(-0.24, 0.56)
    ax.set_xticks([1, 3, 5, 8, 10, 13, 16])
    ax.set_yticks([rows["visual"], rows["text"]])
    ax.set_yticklabels(["Visual", "Text"])
    ax.set_xlabel("Distractor rank  (1 = higher priority)")
    ax.grid(axis="x", color=GRID_C, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.tick_params(axis="x", width=1.0, length=4.5)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)

    save(fig, out_dir, "panel_c_rank_before_after_raincloud")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_trial", type=str, default=str(DEFAULT_PER_TRIAL),
                    help="per_trial.csv written by code/analyze_results.py")
    ap.add_argument("--output_dir", type=str, default=str(HERE))
    ap.add_argument("--backbone", type=str, default="",
                    help="suffix appended to output filenames, e.g. 9b")
    ap.add_argument("--panels", type=str, default="abc",
                    help="subset of panels to render, e.g. 'b' or 'abc'")
    args = ap.parse_args()

    df = load(Path(args.per_trial))
    out_dir = Path(args.output_dir)
    panels = args.panels.lower()

    if "a" in panels:
        panel_a(df, out_dir)
    if "b" in panels:
        panel_b(df, out_dir, args.backbone)
    if "c" in panels:
        panel_c(df, out_dir)

    print(f"Saved figures to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
