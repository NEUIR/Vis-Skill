#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fig. 3b (F4, cross-model): keyword readout probability, one panel, four lines.

Asked "which object name appears most often?", how likely is each model x
modality to answer with exactly the keyword, as a function of exposure k?
Plotted for Qwen3.5-9B and Gemma-4-31B, each with the skill as text and as
image, to show whether the keyword can be read back out of the document /
image in both model families and both modalities.

Data : per-state symbol-readout results of run_symbol_readout.py
       (results/symbol_readout/*.json) for Qwen3.5-9B and Gemma-4-31B. Each
       file stores logP(keyword) at the answer slot; exp() turns it into the
       probability plotted here, in percent, on a linear axis whose range
       follows the data. Grey whiskers = 95% bootstrap CI of the mean over
       states (4000 resamples); the k=8 value is printed at the end of each
       line.
Style: serif Times, horizontal grid #E5EAF2, frameless legend, dpi 600;
       line colours shared with probability-to-behavior/paper_figures/
       plot_fig3c_behavioral_realization.py, modality markers
       Text 'o' / Visual 'D'. The canvas matches
       rasterized-interference/paper_figures/plot_fig3a_rasterized_interference.py:
       figsize (5.8, 5.2), fixed subplots_adjust margins, no tight cropping.

    python plot_fig3b_symbol_readout.py [--qwen_results <dir>] [--gemma_results <dir>] [--output_dir <dir>]
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

LABEL_SIZE  = 14
TICK_SIZE   = 12
LEGEND_SIZE = 10.5
ENDLAB_SIZE = 8.5
# Same canvas as rasterized-interference/paper_figures/
# beeswarm_beta_logp_distractor_9b.png: same figsize, fixed margins and no
# bbox_inches="tight", so both PNGs come out 3480 x 3120 px at 600 dpi.
FIGSIZE     = (5.8, 5.2)
# Margins match plot_fig3a_rasterized_interference.py so the axes box is identical too:
# (0.95-0.18)*5.8 x (0.97-0.14)*5.2 in.
MARGINS     = dict(left=0.18, right=0.95, bottom=0.14, top=0.97)

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
ERR_C  = "#B9C0C9"

# Four low-saturation line colours (dusty blue / lavender / sand / brown).
QWEN_TEXT_C = "#4A6DA7"
QWEN_VIS_C  = "#9B8EC4"
GEMMA_TEXT_C = "#D9A441"
GEMMA_VIS_C  = "#B06A5B"

KS = [0, 4, 8]
HERE = Path(__file__).resolve().parent

# (label, model, modality, colour, marker)
SERIES = [("Qwen3.5-9B · Text",   "qwen",  "text",   QWEN_TEXT_C,  "o"),
          ("Qwen3.5-9B · Visual", "qwen",  "visual", QWEN_VIS_C,   "D"),
          ("Gemma-4-31B · Text",  "gemma", "text",   GEMMA_TEXT_C, "o"),
          ("Gemma-4-31B · Visual", "gemma", "visual", GEMMA_VIS_C, "D")]


# run_symbol_readout.py output dirs (set from --qwen_results / --gemma_results in main()).
RESULTS_DIRS = {"qwen": None, "gemma": None}
DEFAULT_RESULTS_DIRS = {
    "qwen": HERE.parent / "results" / "symbol_readout",
    "gemma": (HERE.parents[3] / "alfworld-gemma-4-31b" / "mechanism"
              / "symbol-encoding" / "results" / "symbol_readout"),
}


def load_probs_from_dir(readout_dir, mod):
    """Read run_symbol_readout.py output: results/symbol_readout/*.json."""
    import glob
    import json
    out = {k: [] for k in KS}
    for f in sorted(glob.glob(str(Path(readout_dir) / "*.json"))):
        try:
            r = json.load(open(f))
        except (ValueError, OSError):
            continue
        if r.get("skipped") or r.get("logp_keyword") is None:
            continue
        if r.get("k") in KS and r.get("modality") == mod:
            out[r["k"]].append(float(np.exp(r["logp_keyword"])))
    return {k: 100.0 * np.asarray(v) for k, v in out.items()}


def load_probs(model, mod):
    """-> {k: array of per-state readout probabilities (%)} for one modality."""
    return load_probs_from_dir(RESULTS_DIRS[model], mod)


def mean_ci(a, n_boot=4000, seed=0):
    a = np.asarray(a, dtype=float)
    rng = np.random.default_rng(seed)
    means = a[rng.integers(0, a.size, (n_boot, a.size))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return a.mean(), a.mean() - lo, hi - a.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", type=str, default=str(HERE))
    ap.add_argument("--from_results", action="store_true",
                    help="accepted for compatibility; the launch.sh outputs are always read")
    ap.add_argument("--qwen_results", type=str, default=str(DEFAULT_RESULTS_DIRS["qwen"]),
                    help="Qwen3.5-9B results/symbol_readout dir")
    ap.add_argument("--gemma_results", type=str, default=str(DEFAULT_RESULTS_DIRS["gemma"]),
                    help="Gemma-4-31B results/symbol_readout dir")
    args = ap.parse_args()
    for m in ("qwen", "gemma"):
        RESULTS_DIRS[m] = Path(getattr(args, f"{m}_results"))
        if not RESULTS_DIRS[m].is_dir():
            raise SystemExit(f"symbol-readout results not found: {RESULTS_DIRS[m]}\n"
                             "Run symbol-encoding/launch.sh for this model first.")
        print(f"  {m}: reading {RESULTS_DIRS[m]}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.subplots_adjust(**MARGINS)

    cache = {}
    top = 0.0
    for label, mdir, mod, color, marker in SERIES:
        probs = cache.setdefault((mdir, mod), load_probs(mdir, mod))
        if not probs[KS[0]].size:
            raise SystemExit(f"No readout data for: {mdir}")
        mean = np.array([probs[k].mean() for k in KS])
        lo, hi = [], []
        for k in KS:
            _m, l, h = mean_ci(probs[k])
            lo.append(l); hi.append(h)
        top = max(top, float(np.max(mean + np.asarray(hi))))
        print(f"  {label:22s}: " +
              "  ".join(f"k={k} {v:8.4f}%" for k, v in zip(KS, mean)))

        ax.errorbar(KS, mean, yerr=[lo, hi], fmt="none", ecolor=ERR_C,
                    elinewidth=0.9, capsize=2.5, capthick=0.9, zorder=2)
        ax.plot(KS, mean, color=color, lw=1.8, solid_capstyle="round", zorder=3)
        ax.scatter(KS, mean, s=32, marker=marker, facecolor=color,
                   edgecolor=color, linewidth=1.0, zorder=4)
        ax.annotate(f"{mean[-1]:.1f}", (KS[-1], mean[-1]),
                    xytext=(6, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=ENDLAB_SIZE, color=color,
                    zorder=5)

    ax.set_xticks(KS)
    ax.set_xlim(-0.4, 9.4)          # room for the k=8 endpoint labels
    ax.set_ylim(0, top * 1.02)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.set_xlabel("Exposure level")
    ax.set_ylabel("Probability of answering\nwith the keyword (%)")
    ax.yaxis.grid(True, color=GRID_C, lw=0.9)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    ax.tick_params(width=1.0, length=4.5)

    ax.legend([Line2D([0], [0], color=c, lw=1.8, marker=m, ms=5.4,
                      mfc=c, mec=c, mew=1.0)
               for _l, _d, _m, c, m in SERIES],
              [l for l, _d, _m, _c, _m2 in SERIES],
              loc="upper left", frameon=False, handlelength=2.2,
              handletextpad=0.5, labelspacing=0.3, borderaxespad=0.4)

    stem = out_dir / "fig3_f4_crossmodel_readout"
    # No bbox_inches="tight": the fixed canvas keeps the pixel size equal to
    # the beeswarm figure.
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)
    print(f"Saved: {stem}.png / .pdf")


if __name__ == "__main__":
    main()
