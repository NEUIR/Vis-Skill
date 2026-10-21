#!/usr/bin/env python3
"""
H1 heatmap (square, blue, noise-robust), one figure per representation.

  * one Text figure and one Visual figure (distractor action only; the
    correct-action side is not drawn)
  * each heatmap box is near-square (set_box_aspect(1))
  * layer / k ticks sit directly on the axes
  * sequential blue colormap
  * noise-robust colour scale: vmin=0, vmax = shared 98th percentile of the
    distractor cells across both modalities, so a few outlier (deep layer, high
    k) cells do not wash the map out. Values above vmax are clipped (shown
    with an over-colour). Cells are the paired-state mean of S(k)-S(0); the
    mean l=31 slope is printed for comparison with lexical-interference.

Input: the per-state logit-lens results of run_layerwise_slope.py
(mechanism/results/layerwise-exposure-slope/per_layer/*.json by default;
override with --from_results PER_LAYER_DIR).

Usage:  python3 plot_fig4_logit_lens.py [--mode sum|norm|both] [--output_dir DIR]
                                        [--from_results [PER_LAYER_DIR]]
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))

NUM_LAYERS = 32
FULL_ATTN = [3, 7, 11, 15, 19, 23, 27, 31]
MODALITIES = ("text", "visual")
KS = (0, 1, 2, 4, 8)
PCTL = 98.0                       # robust colour cap

INK = "#1F2A3A"
GRAY_C = "#8A93A0"

# Font sizes match the lexical-interference figures.
LABEL_SIZE = 13
TICK_SIZE = 11

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.labelsize": LABEL_SIZE,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
    "legend.fontsize": 10.5,
    "axes.linewidth": 1.0,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# --------------------------------------------------------------------------- #
# run_layerwise_slope.py output: results/layerwise-exposure-slope/per_layer/*.json
DEFAULT_RES_DIR = os.path.join(os.path.dirname(os.path.dirname(HERE)), "results",
                               "layerwise-exposure-slope", "per_layer")


def ols_slope(xs, ys):
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    dx = xs - xs.mean(); den = float((dx * dx).sum())
    return float((dx * (ys - ys.mean())).sum() / den) if den else float("nan")


def load_states(mode, results_dir):
    import glob
    import json
    files = sorted(glob.glob(os.path.join(results_dir, "*.json")))
    if not files:
        raise SystemExit(f"no per-state results under {results_dir}; run launch.sh first")
    usable = {}; game_of = {}
    for fp in files:
        d = json.load(open(fp)); sid = d["state_id"]
        game_of[sid] = d.get("game_name") or sid
        per_mod = {}
        for mod in MODALITIES:
            pk = (d.get("per_modality") or {}).get(mod)
            if not pk or any(str(k) not in pk for k in KS):
                continue
            mats = {"distractor": np.full((NUM_LAYERS, len(KS)), np.nan),
                    "correct": np.full((NUM_LAYERS, len(KS)), np.nan)}
            bad = False
            for ki, k in enumerate(KS):
                cell = pk[str(k)]
                if not (cell.get("stem_prefix_ok_c") and cell.get("stem_prefix_ok_d")):
                    bad = True; break
                for side in ("distractor", "correct"):
                    sc = cell["scores"][side]
                    if sc["n_action_tokens"] < 1:
                        bad = True; break
                    denom = sc["n_action_tokens"] if mode == "norm" else 1.0
                    for l in range(NUM_LAYERS):
                        mats[side][l, ki] = float(sc["per_layer_sum"][str(l)]) / denom
                if bad:
                    break
            if not bad:
                per_mod[mod] = mats
        if per_mod:
            usable[sid] = per_mod
    return usable, game_of


def build_from_results(mode, results_dir):
    usable, game_of = load_states(mode, results_dir)
    states = sorted(usable)
    Hd = {}; betaD31 = {}
    ks = np.asarray(KS, float)
    for mod in MODALITIES:
        sts = [s for s in states if mod in usable[s]]
        stack = np.stack([usable[s][mod]["distractor"] for s in sts], axis=0)
        Hd[mod] = np.nanmean(stack - stack[:, :, [0]], axis=0)     # mean of S(k)-S(0)
        betaD31[mod] = float(np.mean([ols_slope(ks, usable[s][mod]["distractor"][31])
                                      for s in sts]))
    return {"Hd": Hd, "betaD31": betaD31, "n_states": len(states),
            "n_games": len(set(game_of[s] for s in states))}


def make_one(H, mod, mode, res, vmax, out_dir):
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_over("#0A2E5C")          # clipped (> vmax) cells
    over = float((H > vmax).sum())

    # Square heatmap; right-side colour bar at the same height.
    figw, figh = 5.4, 5.0
    fig = plt.figure(figsize=(figw, figh))
    ax_l, ax_w, ax_top = 0.15, 0.60, 0.93
    ax_h = ax_w * figw / figh          # square box in physical units
    ax_b = ax_top - ax_h
    ax = fig.add_axes([ax_l, ax_b, ax_w, ax_h])
    cax = fig.add_axes([ax_l + ax_w + 0.018, ax_b, 0.032, ax_h])

    im = ax.imshow(H, aspect="auto", cmap=cmap, vmin=0.0, vmax=vmax, origin="upper",
                   extent=[-0.5, len(KS) - 0.5, NUM_LAYERS - 0.5, -0.5])

    # Ticks/labels directly on the axes.
    ax.set_xticks(range(len(KS)))
    ax.set_xticklabels([str(k) for k in KS])
    ax.set_xlabel("Exposure level  $k$")
    ax.set_yticks([0, 3, 7, 11, 15, 19, 23, 27, 31])
    ax.set_ylim(NUM_LAYERS - 0.5, -0.5)
    ax.set_ylabel("Language-model layer")
    ax.tick_params(length=4, width=1.0)

    # Colour scale: plain vertical bar, same height as the heatmap.
    cb = fig.colorbar(im, cax=cax, extend="neither")
    cb.ax.tick_params(labelsize=TICK_SIZE, length=3)
    cb.outline.set_linewidth(1.0)

    stem = f"figure_h1_{mod}_{mode}"
    fig.savefig(Path(out_dir) / f"{stem}.png", dpi=450, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(Path(out_dir) / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sum", "norm", "both"], default="both")
    ap.add_argument("--output_dir", default=HERE)
    ap.add_argument("--from_results", nargs="?", const=DEFAULT_RES_DIR, default=DEFAULT_RES_DIR,
                    metavar="PER_LAYER_DIR",
                    help="run_layerwise_slope.py output dir (default: %(default)s)")
    args = ap.parse_args()
    for mode in (["sum", "norm"] if args.mode == "both" else [args.mode]):
        res = build_from_results(mode, args.from_results)
        # Shared robust cap across both modalities so the two figures compare.
        allpos = np.concatenate([res["Hd"][m][res["Hd"][m] > 0].ravel() for m in MODALITIES])
        vmax = float(np.nanpercentile(allpos, PCTL))
        vmax = round(vmax, 2)
        bt, bv = res["betaD31"]["text"], res["betaD31"]["visual"]
        print(f"[{mode}] states={res['n_states']} games={res['n_games']} "
              f"vmax(p{int(PCTL)})={vmax}  beta_D[31] text {bt:+.3f}/visual {bv:+.3f}")
        for mod in MODALITIES:
            stem = make_one(res["Hd"][mod], mod, mode, res, vmax, args.output_dir)
            print(f"[{mode}] saved {stem}.{{png,pdf}}")


if __name__ == "__main__":
    main()
