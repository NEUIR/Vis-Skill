#!/usr/bin/env python3
"""
Figure 5: component patching, one figure per model (Qwen3.5-9B and
Gemma-4-31B). Type and canvas settings match
lexical-interference/paper_figures/plot_fig1b_delta_p_distribution.py (serif Times
stack, label 13 / tick 11 / legend 10.5, FIGSIZE (4.15, 3.85), dpi 600 + tight
bbox).

What the scatter shows, per state:
  x = "Exposure effect": preference gap for the distractor action between 8
      and 0 exposures of the skill document (G, in nats)
  y = "Recovered by patch": how much of G returns when the labelled component's
      output from the 8-exposure run is transplanted into the 0-exposure run at
      the action-scoring rows (R)
  grey cloud  = token-mixer transplants; it collapses onto the dotted diagonal
                y=x (the mixer is the only cross-token channel, so swapping all
                of it reproduces the donor exactly)
  clay/violet = MLP transplants, skill as text / as image; their side of the
                dashed zero line is the finding (Qwen below = the MLP brakes
                the bias; Gemma above = the MLP carries it)

No per-state ratio is plotted; the headline percentage (ratio of means,
game-clustered CI) is printed to stdout and reported in the caption.

Input: per_state.csv and analysis.json written by analyze_component_patch.py
(mechanism/results/component-patching/ by default; override with
--from_results RESULTS_DIR).

    python3 plot_fig5_component_patching.py [--output_dir DIR] [--from_results [DIR]]
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABEL_SIZE = 13
TICK_SIZE = 11
LEGEND_SIZE = 10.5
FIGSIZE = (4.15, 3.85)
LEGEND_MS = 5.0

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman",
                   "DejaVu Serif"],
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

TEXT_COLOR = "#C98A63"
VISUAL_COLOR = "#8B87B4"
MIX_COLOR = "#9AA1AB"
ZERO_C = "#B3BAC5"
DIAG_C = "#DDE2E8"

X_LABEL = "Exposure effect"
Y_LABEL = "Recovered by patch"
OUT_NAMES = {"qwen": "figure_b_qwen35_9b", "gemma": "figure_b_gemma4_31b"}
ARCH_ORDER = ["qwen", "gemma"]
MODALITIES = ("text", "visual")
STAGE_ORDER = ("headline", "all", "bands")

# Filled by load_from_results() in main().
G_D, R_D, ARM_SUMMARY = {}, {}, {}


def save(fig, base):
    fig.savefig(base + ".png", dpi=600, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(base + ".pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def cell(arch, mod, arm):
    key = (arch, mod, arm)
    return list(zip(G_D.get(key, []), R_D.get(key, [])))


def draw_figure(arch, base):
    fig, ax = plt.subplots(figsize=FIGSIZE)

    mix = [p for mod in MODALITIES for p in cell(arch, mod, "all_mix")]
    ax.scatter([g for g, _ in mix], [r for _, r in mix], s=6,
               color=MIX_COLOR, alpha=0.45, lw=0, zorder=2)
    mlp = []
    for mod, color in (("text", TEXT_COLOR), ("visual", VISUAL_COLOR)):
        pts = cell(arch, mod, "all_mlp")
        mlp += pts
        ax.scatter([g for g, _ in pts], [r for _, r in pts], s=13,
                   color=color, alpha=0.8, lw=0, zorder=3)

    allg = [g for g, _ in mix + mlp]
    allr = [r for _, r in mix + mlp]
    glo, ghi = np.percentile(allg, [0.5, 99.5])
    rlo, rhi = np.percentile(allr, [0.5, 99.5])
    padg = 0.06 * (ghi - glo)
    padr = 0.06 * (rhi - rlo)
    x0, x1 = glo - padg, ghi + padg
    y0, y1 = min(rlo - padr, x0), max(rhi + padr, x1)
    d = np.array([min(x0, y0), max(x1, y1)])
    ax.plot(d, d, color=DIAG_C, lw=1.0, ls=":", zorder=1)
    ax.axhline(0.0, color=ZERO_C, lw=1.0, ls="--", zorder=1)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    # legend markers are hand-built so all three circles are the SAME size,
    # even though the grey token-mixer dots are drawn smaller on the canvas
    # (they overlap on the diagonal and would otherwise blob together)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", lw=0, ms=LEGEND_MS,
                      markerfacecolor=c, markeredgecolor="none", alpha=a)
               for c, a in ((MIX_COLOR, 0.75), (TEXT_COLOR, 0.9),
                            (VISUAL_COLOR, 0.9))]
    ax.legend(handles, ["Token mixer", "MLP (text)", "MLP (visual)"],
              loc="upper left", frameon=False, handlelength=1.0,
              borderaxespad=0.4, labelspacing=0.3, handletextpad=0.5)
    ax.tick_params(length=3.5, width=1.0)
    fig.tight_layout()
    save(fig, base)
    return base


# --------------------------------------------------------------------------- #
# analyze_component_patch.py output: results/component-patching/{per_state.csv,analysis.json}
DEFAULT_RESULTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "results", "component-patching")


def load_from_results(results_dir):
    import csv
    import json
    ps = os.path.join(results_dir, "per_state.csv")
    an = os.path.join(results_dir, "analysis.json")
    if not os.path.isfile(ps):
        raise SystemExit(f"{ps} not found; run analyze_component_patch.py first")
    by_stage = {}
    with open(ps, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["arm"] not in ("all_mix", "all_mlp"):
                continue
            key = (row["arch"], row["modality"], row["arm"])
            gr = by_stage.setdefault(key, {}).setdefault(row.get("stage", ""), ([], []))
            gr[0].append(float(row["G_d"]))
            gr[1].append(float(row["R_d"]))
    # one stage per cell (same preference order as the summary below), so a
    # state is never drawn twice
    g, r = {}, {}
    for key, stages in by_stage.items():
        stage = next((st for st in STAGE_ORDER if st in stages), next(iter(stages)))
        g[key], r[key] = stages[stage]
    summ = {}
    if os.path.isfile(an):
        with open(an, encoding="utf-8") as f:
            blocks = {(b["arch"], b["stage"], b["modality"]): b
                      for b in json.load(f).get("blocks", [])}
        for (arch, mod, arm) in g:
            for stage in STAGE_ORDER:
                a = (blocks.get((arch, stage, mod)) or {}).get("arms", {}).get(arm)
                if a and a.get("n"):
                    summ[(arch, mod, arm)] = (a["n"], a["recovered_distractor_pct"],
                                              a["ci_lo_pct"], a["ci_hi_pct"])
                    break
    return g, r, summ


def main():
    global G_D, R_D, ARM_SUMMARY
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--from_results", nargs="?", const=DEFAULT_RESULTS, default=DEFAULT_RESULTS,
                    metavar="RESULTS_DIR",
                    help="dir with per_state.csv / analysis.json written by "
                         "analyze_component_patch.py (default: %(default)s)")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    G_D, R_D, ARM_SUMMARY = load_from_results(args.from_results)

    for arch in ARCH_ORDER:
        if not any(k[0] == arch for k in G_D):
            print(f"[fig b] no cells for {arch}; skipped")
            continue
        base = draw_figure(arch, os.path.join(args.output_dir, OUT_NAMES[arch]))
        print(f"[fig b] wrote {base}.png / .pdf")

    for arch in sorted(ARCH_ORDER):
        for mod in MODALITIES:
            if ((arch, mod, "all_mlp") not in ARM_SUMMARY
                    or (arch, mod, "all_mix") not in ARM_SUMMARY):
                continue
            _, mx_pct, _, _ = ARM_SUMMARY[(arch, mod, "all_mix")]
            n, ml_pct, lo, hi = ARM_SUMMARY[(arch, mod, "all_mlp")]
            print(f"  {arch:5s} {mod:6s} n={n:3d}  mixer "
                  f"{mx_pct:+6.1f}%  mlp "
                  f"{ml_pct:+6.1f} [{lo:+6.1f},{hi:+6.1f}]")


if __name__ == "__main__":
    main()
