#!/usr/bin/env python3
"""
symbol-encoding / F3 analysis — identity vs count, as a rescue experiment.

Three slopes per state, all on k in {0,4,8}:
    native        doc lowercase   , action lowercase   (identity matched, 1 token)
    upper/altcase doc recased     , action lowercase   (identity BROKEN)     [F1]
    match_*       doc recased     , action recased     (identity RESTORED)   [F3]

RESCUE FRACTION = (match - broken) / (native - broken), per state, paired.
    ~1.0  -> the F1 casing cost was ENTIRELY the document symbol no longer being
             the same symbol the model must emit. Exposure works by matching a
             discrete symbol, not by "being a crisp/short token": match_altcase
             still has a 4-6 token mangled keyword and yet recovers.
    ~0.0  -> the cost was the mangled form itself (crispness / segmentation);
             re-pairing does not help.
    in between -> both, and the split is the number reported.

Each match arm is anchored on its OWN k=0 cell (the scored action string differs,
so absolute logP is not comparable across arms — only slopes are).

Run: python3 analyze_case_matched.py
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
PAIRS = [("upper", "match_upper"), ("altcase", "match_altcase")]
# PRIMARY contrast: same scored string, document identity varied.
IDENTITY_PAIRS = [("mismatch_upper", "match_upper"),
                  ("mismatch_altcase", "match_altcase")]
OUT = os.path.join(HERE, "results", "case_matched_analysis.json")


def clustered_ci(by_group, n_boot=4000, seed=0):
    groups = list(by_group)
    if not groups:
        return None, None, None
    rng = np.random.default_rng(seed)
    means = [float(np.mean(by_group[g])) for g in groups]
    boots = [np.mean([means[i] for i in rng.integers(0, len(groups), len(groups))])
             for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(np.mean(means)), float(lo), float(hi)


def load_dir(d, own_anchor: bool):
    """-> (slopes[arm][sid], games, extra). own_anchor=False shares native's k=0."""
    cells = defaultdict(dict)
    games, ident = {}, defaultdict(list)
    for f in glob.glob(os.path.join(d, "*.json")):
        r = json.load(open(f))
        if r.get("skipped") or "logp_distractor" not in r or r["k"] not in KS:
            continue
        cells[(r["state_id"], r["arm"])][r["k"]] = r
        games[r["state_id"]] = r.get("game_name") or r["state_id"]
        if r["k"] > 0:
            ident[r["arm"]].append(bool(r.get("meta", {}).get("identity_restored", True)))
    anchor = {sid: c[0] for (sid, arm), c in cells.items()
              if arm == "native" and 0 in c} if not own_anchor else {}
    slopes = defaultdict(dict)
    for (sid, arm), byk in cells.items():
        d_ = {k: r["logp_distractor"] for k, r in byk.items()}
        c_ = {k: r["logp_correct"] for k, r in byk.items()}
        if 0 not in d_ and sid in anchor:
            d_[0] = anchor[sid]["logp_distractor"]
            c_[0] = anchor[sid]["logp_correct"]
        if len(d_) == 3:
            ks = sorted(d_)
            slopes[arm][sid] = (float(np.polyfit(ks, [d_[k] for k in ks], 1)[0]),
                                float(np.polyfit(ks, [c_[k] for k in ks], 1)[0]))
    return slopes, games, ident


def main():
    t_dir = os.path.join(HERE, "results", "text_casing")
    m_dir = os.path.join(HERE, "results", "case_matched")
    if not os.path.isdir(m_dir) or not glob.glob(os.path.join(m_dir, "*.json")):
        print("[F3] no case_matched results yet")
        return
    T, gT, _ = load_dir(t_dir, own_anchor=False)
    M, gM, ident = load_dir(m_dir, own_anchor=True)
    games = {**gT, **gM}
    for arm, v in sorted(ident.items()):
        if v and not all(v):
            print(f"  WARNING {arm}: keyword absent from the distractor action in "
                  f"{v.count(False)}/{len(v)} cells -> arm is a no-op there")

    print(f"[F3] native n={len(T.get('native', {}))}  "
          + "  ".join(f"{a}={len(M.get(a,{}))}" for a in sorted(M)))
    print(f"\n{'='*92}\nIDENTITY RESCUE  (paired per state; k={KS})\n{'='*92}")
    print(f"  {'contrast':16s} {'n':>4s} {'native':>8s} {'broken':>8s} {'matched':>8s} "
          f"{'rescue [95% CI]':>26s} {'sign':>11s}")
    out = {"ks": KS, "contrasts": {}}
    for broken, matched in PAIRS:
        if broken not in T or matched not in M or "native" not in T:
            continue
        common = sorted(set(T["native"]) & set(T[broken]) & set(M[matched]))
        if not common:
            continue
        nat = np.array([T["native"][s][0] for s in common])
        brk = np.array([T[broken][s][0] for s in common])
        mat = np.array([M[matched][s][0] for s in common])
        # rescue on the AGGREGATE denominator; a per-state ratio explodes when a
        # state's (native - broken) is near zero, so the ratio is formed from the
        # bootstrap of the two paired differences, not from per-state ratios.
        num = mat - brk
        den = nat - brk
        rng = np.random.default_rng(0)
        boots = []
        for _ in range(4000):
            i = rng.integers(0, len(common), len(common))
            d = den[i].mean()
            if abs(d) > 1e-9:
                boots.append(num[i].mean() / d)
        r_point = float(num.mean() / den.mean()) if abs(den.mean()) > 1e-9 else float("nan")
        rlo, rhi = np.percentile(boots, [2.5, 97.5])
        npos, nneg, p = sign_test(list(num))
        print(f"  {broken:>7s}->{matched:<8s} {len(common):4d} {nat.mean():8.4f} "
              f"{brk.mean():8.4f} {mat.mean():8.4f} "
              f"{r_point:8.2f} [{rlo:7.2f},{rhi:7.2f}] {npos:4d}+/{nneg:4d}-")
        mn, lon, hin = bootstrap_mean_ci(list(num))
        md, lod, hid = bootstrap_mean_ci(list(den))
        byg = defaultdict(list)
        for s, v in zip(common, num):
            byg[games.get(s, s)].append(float(v))
        cm, clo, chi = clustered_ci(byg)
        print(f"  {'':16s}      recovered slope (matched-broken) {mn:+.4f} "
              f"[{lon:+.4f},{hin:+.4f}]  game-clustered [{clo:+.4f},{chi:+.4f}]")
        print(f"  {'':16s}      full casing cost (native-broken) {md:+.4f} "
              f"[{lod:+.4f},{hid:+.4f}]")
        print(f"  {'':16s}      logP(correct) slope matched {np.mean([M[matched][s][1] for s in common]):+.4f}"
              f"   native {np.mean([T['native'][s][1] for s in common]):+.4f}  (control ~0)")
        out["contrasts"][f"{broken}->{matched}"] = {
            "n": len(common), "slope_native": float(nat.mean()),
            "slope_broken": float(brk.mean()), "slope_matched": float(mat.mean()),
            "rescue_fraction": r_point, "rescue_ci": [float(rlo), float(rhi)],
            "recovered": [mn, lon, hin], "recovered_clustered": [cm, clo, chi],
            "full_cost": [md, lod, hid], "n_pos": npos, "n_neg": nneg, "sign_p": p,
            "slope_cor_matched": float(np.mean([M[matched][s][1] for s in common])),
            "slope_cor_native": float(np.mean([T["native"][s][1] for s in common]))}

    print("\n  CAVEAT: the rescue ratio above is CONFOUNDED and is reported only for")
    print("  the record. A recased action has more tokens and a far lower prior, so its")
    print("  k=0 baseline is much lower and its dynamic range much larger than native's;")
    print("  slopes are not comparable across different scored strings. The clean")
    print("  contrast is below, where the scored string is held FIXED.")

    # ── PRIMARY: identity effect with the scored string held fixed ──────
    print(f"\n{'='*92}")
    print("IDENTITY EFFECT, scored string held FIXED (this is the clean test)")
    print(f"{'='*92}")
    print("  Both arms score the SAME recased action (same tokens, same prior, same k=0")
    print("  anchor). Only the DOCUMENT differs: match_* contains the identical recased")
    print("  symbol, mismatch_* contains the native lowercase form. So the difference is")
    print("  purely 'is the document symbol the symbol I must emit'.\n")
    print(f"  {'contrast':26s} {'n':>4s} {'mismatch':>9s} {'match':>9s} "
          f"{'identity gain [95% CI]':>28s} {'sign':>11s}")
    ident_out = {}
    for mis, mat in IDENTITY_PAIRS:
        if mis not in M or mat not in M:
            continue
        common = sorted(set(M[mis]) & set(M[mat]))
        if not common:
            continue
        a = np.array([M[mis][s][0] for s in common])
        b = np.array([M[mat][s][0] for s in common])
        d = list(b - a)
        m, lo, hi = bootstrap_mean_ci(d)
        byg = defaultdict(list)
        for s, x in zip(common, b - a):
            byg[games.get(s, s)].append(float(x))
        cm, clo, chi = clustered_ci(byg)
        npos, nneg, p = sign_test(d)
        print(f"  {mis:>16s}->{mat:<9s} {len(d):4d} {a.mean():9.4f} {b.mean():9.4f} "
              f"{m:+9.4f} [{lo:+8.4f},{hi:+8.4f}] {npos:4d}+/{nneg:4d}-")
        print(f"  {'':26s}      game-clustered [{clo:+.4f},{chi:+.4f}]  p={p:.3g}")
        ident_out[f"{mis}->{mat}"] = {
            "n": len(d), "slope_mismatch": float(a.mean()), "slope_match": float(b.mean()),
            "identity_gain": [m, lo, hi], "clustered": [cm, clo, chi],
            "n_pos": npos, "n_neg": nneg, "sign_p": p}
    if ident_out:
        print("\n  READING: gain >> 0 => exposure acts by SYMBOL IDENTITY -- k occurrences")
        print("  help only insofar as they are the very token to be emitted. gain ~ 0 =>")
        print("  exposure works through the word's meaning, not its surface identity.")
    out["identity_fixed_string"] = ident_out
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\n[F3] wrote {OUT}")


if __name__ == "__main__":
    main()
