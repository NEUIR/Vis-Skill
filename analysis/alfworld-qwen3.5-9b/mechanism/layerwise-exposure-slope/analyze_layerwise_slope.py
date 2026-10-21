#!/usr/bin/env python3
"""
Analysis for the layer-wise exposure-slope experiment.

Per state, per layer l, per modality, we have s_l(a_d; k) and s_l(a_c; k).
Three slopes are fitted per (state, layer, modality) by OLS over k:

    beta_M[l]  slope of  M_l(k) = s_l(a_d;k) - s_l(a_c;k)     <- the headline
    beta_D[l]  slope of  s_l(a_d;k)                            <- GATE + decomposition
    beta_C[l]  slope of  s_l(a_c;k)                            <- control: must stay ~0

Aggregation NEVER averages per-state ratios or normalises per state; slopes are
fitted per state (same k support for every state) and then averaged, with a
GAME-CLUSTERED bootstrap CI (resample games, average within game, then across)
because the 165 states come from only ~64 games.  Sign test is secondary.

GATE (hard requirement before anything downstream is interpretable)
    The lens at l=31 IS the model's own final norm + lm_head, so
    s_31(a_d; k) must equal LI's `logp_distractor_action`.  Therefore
    beta_D[31] must reproduce the published anchors:
        text   0.741 on k in {0,1,2,4,8}   /  0.847 on k in {0,4,8}
        visual 0.552 on k in {0,1,2,4,8}   /  0.591 on k in {0,4,8}
    If it does not, the prompt assembly or the stem fix is wrong and the
    layer-wise curve is meaningless.  Reported as PASS / CHECK, never silenced.

Usage:  python3 analyze_layerwise_slope.py [--k_subset 0 1 2 4 8] [--out analysis_k.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MECH_ROOT = os.path.dirname(HERE)
RES_DIR = os.path.join(MECH_ROOT, "results", "layerwise-exposure-slope", "per_layer")

NUM_LAYERS = 32
FULL_ATTN = [3, 7, 11, 15, 19, 23, 27, 31]
DELTA_NET = [l for l in range(NUM_LAYERS) if l not in FULL_ATTN]
MODALITIES = ("text", "visual")

N_BOOT = 4000
SEED = 0

# published anchors for the GATE
ANCHORS = {
    ("text", (0, 1, 2, 4, 8)): 0.741,
    ("text", (0, 4, 8)): 0.847,
    ("visual", (0, 1, 2, 4, 8)): 0.552,
    ("visual", (0, 4, 8)): 0.591,
}
GATE_TOL = 0.06


# ---------------------------------------------------------------------------
def ols_slope(xs, ys):
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    if xs.size < 2:
        return float("nan")
    dx = xs - xs.mean()
    den = float((dx * dx).sum())
    if den == 0:
        return float("nan")
    return float((dx * (ys - ys.mean())).sum() / den)


def clustered_ci(triples, n_boot=N_BOOT, seed=SEED):
    """triples: list of (game, value). Returns (point, lo, hi, n, n_games)."""
    if not triples:
        return (float("nan"),) * 3 + (0, 0)
    by_game = defaultdict(list)
    for g, v in triples:
        by_game[g].append(v)
    games = sorted(by_game)

    def stat(sample):
        return float(np.mean([np.mean(by_game[g]) for g in sample]))

    point = stat(games)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(len(games), size=len(games), replace=True)
        boots[b] = stat([games[i] for i in pick])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi), len(triples), len(games)


def sign_test(vals):
    v = np.asarray(vals, dtype=np.float64)
    v = v[np.isfinite(v)]
    pos, neg = int((v > 0).sum()), int((v < 0).sum())
    n = pos + neg
    if n == 0:
        return {"pos": 0, "neg": 0, "p": float("nan")}
    k = min(pos, neg)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return {"pos": pos, "neg": neg, "p": min(1.0, p)}


# ---------------------------------------------------------------------------
def load(k_subset):
    """-> usable[state_id] = {modality: {layer: {side: [s(k) for k in subset]}}},
    plus game_of[state_id] and a dropped counter."""
    usable, dropped = {}, defaultdict(int)
    game_of = {}
    files = sorted(glob.glob(os.path.join(RES_DIR, "*.json")))
    for fp in files:
        if os.path.basename(fp) == "errors.log":
            continue
        d = json.load(open(fp))
        sid = d["state_id"]
        game_of[sid] = d.get("game_name") or sid
        per_mod = {}
        ok_state = True
        for mod in MODALITIES:
            pk = (d.get("per_modality") or {}).get(mod)
            if not pk:
                continue
            # completeness + QA gates for this modality
            if any(str(k) not in pk for k in k_subset):
                dropped[f"{mod}:missing_k"] += 1
                ok_state = False
                break
            bad = False
            for k in k_subset:
                cell = pk[str(k)]
                if not (cell.get("stem_prefix_ok_c") and cell.get("stem_prefix_ok_d")):
                    dropped[f"{mod}:stem_prefix"] += 1
                    bad = True
                    break
                for side in ("distractor", "correct"):
                    if cell["scores"][side]["n_action_tokens"] < 1:
                        dropped[f"{mod}:empty_action"] += 1
                        bad = True
                        break
                if bad:
                    break
            if bad:
                ok_state = False
                break
            per_mod[mod] = {
                l: {side: [pk[str(k)]["scores"][side]["per_layer_sum"][str(l)]
                           for k in k_subset]
                    for side in ("distractor", "correct")}
                for l in range(NUM_LAYERS)
            }
        if not ok_state or not per_mod:
            continue
        usable[sid] = per_mod
    return usable, dict(dropped), game_of, files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k_subset", nargs="*", type=int, default=[0, 1, 2, 4, 8])
    ap.add_argument("--out", default=os.path.join(HERE, "analysis_k.json"))
    ap.add_argument("--csv", default=os.path.join(HERE, "layerwise_slopes.csv"))
    args = ap.parse_args()

    k_subset = tuple(sorted(args.k_subset))
    ks = np.asarray(k_subset, dtype=np.float64)

    usable, dropped, game_of, files = load(k_subset)
    states = sorted(usable)
    print(f"loaded {len(files)} files | usable states {len(states)} | "
          f"dropped {dropped or '{}'} | k_subset {k_subset}")
    if not states:
        raise SystemExit("no usable states — run run_layerwise_slope.py first")

    games = [game_of[s] for s in states]
    out = {"k_subset": list(k_subset), "n_states": len(states),
           "n_games": len(set(games)), "dropped": dropped,
           "layers": {"full_attn": FULL_ATTN, "delta_net": DELTA_NET},
           "per_layer": {}, "gate": {}, "band_summary": {}, "text_vs_visual": {}}

    # ---- per (modality, layer) slopes ------------------------------------
    # beta[mod][l] = list over states
    beta = {m: {"M": [], "D": [], "C": []} for m in MODALITIES}
    for mod in MODALITIES:
        sts = [s for s in states if mod in usable[s]]
        for l in range(NUM_LAYERS):
            bM, bD, bC = [], [], []
            for s in sts:
                sd = np.asarray(usable[s][mod][l]["distractor"], dtype=np.float64)
                sc = np.asarray(usable[s][mod][l]["correct"], dtype=np.float64)
                bD.append(ols_slope(ks, sd))
                bC.append(ols_slope(ks, sc))
                bM.append(ols_slope(ks, sd - sc))
            beta[mod]["M"].append((sts, bM))
            beta[mod]["D"].append((sts, bD))
            beta[mod]["C"].append((sts, bC))

    for l in range(NUM_LAYERS):
        entry = {"layer": l, "type": "full_attn" if l in FULL_ATTN else "delta_net"}
        for mod in MODALITIES:
            for q in ("M", "D", "C"):
                sts, vals = beta[mod][q][l]
                trip = [(game_of[s], v) for s, v in zip(sts, vals) if np.isfinite(v)]
                pt, lo, hi, n, ng = clustered_ci(trip)
                st = sign_test([v for _, v in trip])
                entry[f"{q}_{mod}"] = {"beta": pt, "ci": [lo, hi], "n": n,
                                       "n_games": ng, "sign": st}
        # paired text - visual on M, over states present in both
        both = [s for s in states if "text" in usable[s] and "visual" in usable[s]]
        if both:
            i_t = [s for s in states if "text" in usable[s]]
            i_v = [s for s in states if "visual" in usable[s]]
            mt = dict(zip(i_t, beta["text"]["M"][l][1]))
            mv = dict(zip(i_v, beta["visual"]["M"][l][1]))
            dif = [(game_of[s], mt[s] - mv[s]) for s in both
                   if np.isfinite(mt[s]) and np.isfinite(mv[s])]
            pt, lo, hi, n, ng = clustered_ci(dif)
            st = sign_test([v for _, v in dif])
            entry["M_text_minus_visual"] = {"delta": pt, "ci": [lo, hi], "n": n,
                                            "n_games": ng, "sign": st}
        out["per_layer"][str(l)] = entry

    # ---- GATE ------------------------------------------------------------
    print("\n" + "=" * 78)
    print("GATE — beta_D at layer 31 must reproduce the published LI anchors")
    for mod in MODALITIES:
        e = out["per_layer"]["31"][f"D_{mod}"]
        anchor = ANCHORS.get((mod, k_subset))
        if anchor is None:
            verdict = "NO ANCHOR for this k_subset"
            ok = None
        else:
            ok = abs(e["beta"] - anchor) <= GATE_TOL
            verdict = ("PASS" if ok else "CHECK") + \
                      f" (|{e['beta']:.4f} - {anchor:.3f}| = {abs(e['beta']-anchor):.4f}, tol {GATE_TOL})"
        out["gate"][mod] = {"beta_D_layer31": e["beta"], "ci": e["ci"],
                            "anchor": anchor, "verdict": verdict, "pass": ok}
        print(f"  {mod:7s} beta_D[31] = {e['beta']:+.4f} [{e['ci'][0]:+.4f},{e['ci'][1]:+.4f}]"
              f"  anchor {anchor}  -> {verdict}")
        # control side must stay near zero
        ec = out["per_layer"]["31"][f"C_{mod}"]
        print(f"          beta_C[31] = {ec['beta']:+.4f}  (control; published ~ -0.03)")

    # ---- band summary: DeltaNet vs FullAttn ------------------------------
    print("\n" + "=" * 78)
    print("BAND SUMMARY — mean beta_M by layer type")
    for mod in MODALITIES:
        for band, layers in (("delta_net", DELTA_NET), ("full_attn", FULL_ATTN)):
            vals = [(game_of[s], v)
                    for l in layers
                    for s, v in zip(beta[mod]["M"][l][0], beta[mod]["M"][l][1])
                    if np.isfinite(v)]
            pt, lo, hi, n, ng = clustered_ci(vals)
            out["band_summary"][f"{mod}_{band}"] = {"mean_beta_M": pt,
                                                    "ci": [lo, hi], "n_layers": len(layers)}
            print(f"  {mod:7s} {band:10s} mean beta_M = {pt:+.4f} [{lo:+.4f},{hi:+.4f}]")

    # ---- where does M first become reliably positive? --------------------
    for mod in MODALITIES:
        onset = None
        for l in range(NUM_LAYERS):
            e = out["per_layer"][str(l)][f"M_{mod}"]
            if e["ci"][0] > 0:
                onset = l
                break
        out.setdefault("onset", {})[mod] = onset
        print(f"  {mod:7s} first layer with beta_M CI excluding 0: L{onset}")

    # ---- CSV for plotting ------------------------------------------------
    import csv
    with open(args.csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "layer_type",
                    "beta_M_text", "lo_M_text", "hi_M_text",
                    "beta_M_visual", "lo_M_visual", "hi_M_visual",
                    "beta_D_text", "beta_D_visual",
                    "beta_C_text", "beta_C_visual",
                    "delta_M_text_minus_visual", "lo_delta", "hi_delta"])
        for l in range(NUM_LAYERS):
            e = out["per_layer"][str(l)]
            row = [l, e["type"]]
            for mod in MODALITIES:
                m = e[f"M_{mod}"]
                row += [round(m["beta"], 6), round(m["ci"][0], 6), round(m["ci"][1], 6)]
            for q in ("D", "C"):
                for mod in MODALITIES:
                    row.append(round(e[f"{q}_{mod}"]["beta"], 6))
            d = e.get("M_text_minus_visual")
            row += ([round(d["delta"], 6), round(d["ci"][0], 6), round(d["ci"][1], 6)]
                    if d else ["", "", ""])
            w.writerow(row)
    print(f"\nwrote {args.csv}")

    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
