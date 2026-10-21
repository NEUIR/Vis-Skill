#!/usr/bin/env python3
"""
symbol-encoding / F5 — does MODALITY do anything beyond changing how strongly the keyword
symbol got instantiated?

The behavioural gap to explain: action-logP exposure slope is 0.848 in text and
0.591 in visual on the same skill content. F4 then found that the SYMBOL itself
is recovered from pixels at statistically the same rate per repetition (readout
slope 0.850 vs 0.817, T-V n.s.) but to a LOWER absolute level (-2.31 vs -3.71 at
k=8). So "the pixels never become a symbol" is refuted, and the live account is
"same saturating curve, visual sits lower on it".

That account makes a sharp, falsifiable prediction: pool text and visual states
into ONE scatter of action slope against readout level, and the MODALITY term
should add nothing. If a modality dummy survives adjustment for readout level,
then something about the visual channel matters over and above how available the
symbol is, and the single-curve story is wrong.

Reported
--------
  1. Raw paired modality gap in action slope (the thing to be explained).
  2. Raw paired modality gap in readout level (the proposed explanation).
  3. Pooled OLS: action_slope ~ 1 + readout + modality_dummy.
     The modality coefficient is the residual gap. Game-clustered bootstrap CI.
  4. % of the raw gap accounted for, and the same fit with readout SLOPE as the
     mediator instead of readout LEVEL (neither choice is defended a priori).

CAVEAT, printed and meant: this is observational mediation. Action slope and
readout come from forward passes on the SAME documents, so state-level nuisance
variance (prompt length, task family, base difficulty) is shared, and adjusting
for a mediator measured post-treatment can bias the direct effect in either
direction. It cannot establish that raising visual readout would raise the
visual action slope. The causal version manipulates recovery quality (casing
arms, E2 blur) while measuring BOTH readout and action slope, and checks that
the manipulated points land on the same curve. This script sets that up; it does
not substitute for it.

Run: python3 analyze_single_curve.py
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MECH_ROOT = os.path.dirname(HERE)
sys.path.insert(0, MECH_ROOT)
from lib.stats import bootstrap_mean_ci, sign_test  # noqa: E402

KS = [0, 4, 8]
RO = os.path.join(HERE, "results", "symbol_readout")
OUT = os.path.join(HERE, "results", "single_curve_analysis.json")
MODS = ("text", "visual")


def fit(d):
    ks = sorted(d)
    return float(np.polyfit(ks, [d[k] for k in ks], 1)[0]) if len(ks) >= 2 else None


def load_action():
    """F1/F2b native-arm action-logP slope per (modality, state)."""
    out, games = {}, {}
    for mod, sub in (("text", "text_casing"), ("visual", "visual_casing")):
        byk = defaultdict(dict)
        for f in glob.glob(os.path.join(HERE, "results", sub, "*.json")):
            r = json.load(open(f))
            if r.get("skipped") or r.get("arm") != "native" or "logp_distractor" not in r:
                continue
            byk[r["state_id"]][r["k"]] = r["logp_distractor"]
            games[r["state_id"]] = r.get("game_name") or r["state_id"]
        out[mod] = {s: fit(v) for s, v in byk.items() if len(v) == 3}
    return out, games


def load_readout():
    """logp_keyword by (modality, state, k) -> level at k=8 and slope over KS."""
    raw = defaultdict(lambda: defaultdict(dict))
    for f in glob.glob(os.path.join(RO, "*.json")):
        r = json.load(open(f))
        if r.get("skipped"):
            continue
        raw[r["modality"]][r["state_id"]][r["k"]] = r["logp_keyword"]
    lvl, slp = defaultdict(dict), defaultdict(dict)
    for mod in raw:
        for sid, byk in raw[mod].items():
            if set(KS) <= set(byk):
                lvl[mod][sid] = byk[8]
                slp[mod][sid] = fit({k: byk[k] for k in KS})
    return lvl, slp


def ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan")
    return beta, r2


def clustered_coef_ci(rows, mediator, n_boot=4000, seed=0):
    """Resample GAMES, refit, return the modality coefficient interval.

    rows: list of (game, action_slope, mediator_value, is_text)
    """
    by_game = defaultdict(list)
    for g, a, m, t in rows:
        by_game[g].append((a, m, t))
    games = list(by_game)
    rng = np.random.default_rng(seed)
    coefs = []
    for _ in range(n_boot):
        pick = [games[i] for i in rng.integers(0, len(games), len(games))]
        pts = [p for g in pick for p in by_game[g]]
        A = np.array([[1.0, m, t] for _, m, t in pts])
        y = np.array([a for a, _, _ in pts])
        try:
            b, _ = ols(A, y)
            coefs.append(float(b[2]))
        except np.linalg.LinAlgError:
            pass
    lo, hi = np.percentile(coefs, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    act, games = load_action()
    lvl, slp = load_readout()
    sids = sorted(set(act["text"]) & set(act["visual"])
                  & set(lvl["text"]) & set(lvl["visual"]))
    print(f"[F5] states with action slope AND readout in BOTH modalities: n={len(sids)}")
    if len(sids) < 20:
        print("[F5] not enough joined states")
        return

    res = {"n": len(sids)}

    # ── 1/2. the gap to explain, and the proposed explanation ───────────
    print(f"\n{'='*84}\nRAW PAIRED MODALITY GAPS (text - visual, per state)\n{'='*84}")
    print(f"  {'quantity':22s} {'text':>9s} {'visual':>9s} {'T-V':>9s} "
          f"{'95% CI':>22s} {'sign':>12s}")
    for label, tbl in (("action slope", act), ("readout level k=8", lvl),
                       ("readout slope", slp)):
        d = [tbl["text"][s] - tbl["visual"][s] for s in sids]
        m, lo, hi = bootstrap_mean_ci(d)
        pos, neg, p = sign_test(d)
        print(f"  {label:22s} {np.mean([tbl['text'][s] for s in sids]):9.4f} "
              f"{np.mean([tbl['visual'][s] for s in sids]):9.4f} {m:9.4f} "
              f"[{lo:+.4f},{hi:+.4f}]".rjust(9) + f"  {pos:4d}+/{neg:4d}- p={p:.3g}")
        res[label] = {"text": float(np.mean([tbl["text"][s] for s in sids])),
                      "visual": float(np.mean([tbl["visual"][s] for s in sids])),
                      "gap": m, "ci": [lo, hi], "sign": [pos, neg, p]}

    # ── 3. pooled single-curve fit ──────────────────────────────────────
    for mname, med in (("readout LEVEL k=8", lvl), ("readout SLOPE", slp)):
        print(f"\n{'='*84}\nSINGLE-CURVE TEST, mediator = {mname}\n{'='*84}")
        rows = [(games.get(s, s), act[m][s], med[m][s], 1.0 if m == "text" else 0.0)
                for s in sids for m in MODS]
        X = np.array([[1.0, r[2], r[3]] for r in rows])
        y = np.array([r[1] for r in rows])

        # modality-only model: the coefficient IS the raw gap
        b0, r2_0 = ols(np.array([[1.0, r[3]] for r in rows]), y)
        # mediator only
        b1, r2_1 = ols(np.array([[1.0, r[2]] for r in rows]), y)
        # both
        b2, r2_2 = ols(X, y)
        lo, hi = clustered_coef_ci(rows, med)

        raw_gap = float(b0[1])
        adj_gap = float(b2[2])
        print(f"  modality only        : gap = {raw_gap:+.4f}   R2={r2_0:.3f}")
        print(f"  {mname:20s} : beta = {float(b1[1]):+.4f}   R2={r2_1:.3f}")
        print(f"  both                 : gap = {adj_gap:+.4f}   "
              f"game-clustered CI [{lo:+.4f},{hi:+.4f}]   R2={r2_2:.3f}")
        share = (1.0 - adj_gap / raw_gap) * 100 if abs(raw_gap) > 1e-9 else float("nan")
        print(f"  -> {share:.1f}% of the raw modality gap is accounted for by {mname}")
        if lo <= 0 <= hi:
            verdict = ("SINGLE CURVE: modality adds nothing once the symbol's "
                       "availability is held fixed")
        elif adj_gap > 0:
            verdict = ("MODALITY SURVIVES: text still helps more at equal symbol "
                       "availability -> the channel matters beyond availability")
        else:
            verdict = "SIGN FLIP: visual helps more at equal availability"
        print(f"  -> {verdict}")
        res[f"fit_{mname}"] = {"raw_gap": raw_gap, "adj_gap": adj_gap,
                              "clustered_ci": [lo, hi], "share_pct": share,
                              "beta_mediator": float(b1[1]),
                              "r2": {"modality": r2_0, "mediator": r2_1, "both": r2_2},
                              "verdict": verdict}

    print(f"\n{'='*84}\nCAVEAT\n{'='*84}")
    print("  Observational mediation. Action slope and readout come from forward")
    print("  passes on the SAME documents, so state-level nuisance variance is")
    print("  shared, and adjusting for a post-treatment mediator can bias the")
    print("  direct effect either way. This cannot show that RAISING visual")
    print("  readout would raise the visual action slope. The causal test")
    print("  manipulates recovery quality (casing arms / E2 blur) while measuring")
    print("  BOTH readout and action slope, then checks single-curve collapse.")

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(f"\n[F5] wrote {OUT}")


if __name__ == "__main__":
    main()
