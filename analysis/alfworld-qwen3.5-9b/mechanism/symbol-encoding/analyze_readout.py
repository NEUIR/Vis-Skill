#!/usr/bin/env python3
"""
symbol-encoding / F4 analysis — is the keyword SYMBOL less available on the language side
when it arrived as pixels, and does that track the behavioural exposure effect?

Two readouts per (state, k, modality):
  logp_keyword          logP of the native lowercase keyword at the answer slot
  margin                logp_keyword - logp_control  (frequency-matched word)

Reported:
  1. Mean readout by (modality, k) — the raw dose curve.
  2. Readout SLOPE per state: keyword-logP slope on k in {0,4,8} and margin slope
     on k in {4,8} (no control exists at k=0, so both variants are shown rather
     than defending one choice).
  3. Text vs visual readout slope, PAIRED per state, bootstrap + game-clustered
     CI + sign test. This is the direct measurement of "what the keyword got
     encoded as".
  4. MEDIATION, observational: within each modality, correlation across states
     between readout slope and the F1/F2b action-logP exposure slope. Reported
     with the caveat printed, never as a causal claim.

Run: python3 analyze_readout.py
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
from lib.stats import bootstrap_mean_ci, sign_test, pearson, spearman  # noqa: E402

KS = [0, 4, 8]
RO = os.path.join(HERE, "results", "symbol_readout")
OUT = os.path.join(HERE, "results", "readout_analysis.json")


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


def fit(d):
    ks = sorted(d)
    return float(np.polyfit(ks, [d[k] for k in ks], 1)[0]) if len(ks) >= 2 else None


def action_slopes():
    """F1 native + F2b native action-logP slopes, for the mediation check."""
    out = {}
    for mod, d in (("text", "text_casing"), ("visual", "visual_casing")):
        cells = defaultdict(dict)
        for f in glob.glob(os.path.join(HERE, "results", d, "*.json")):
            r = json.load(open(f))
            if r.get("skipped") or r.get("arm") != "native" or "logp_distractor" not in r:
                continue
            cells[r["state_id"]][r["k"]] = r["logp_distractor"]
        out[mod] = {s: fit(v) for s, v in cells.items() if len(v) == 3}
    return out


def main():
    if not os.path.isdir(RO) or not glob.glob(os.path.join(RO, "*.json")):
        print("[F4] no readout results yet")
        return
    raw = defaultdict(lambda: defaultdict(dict))   # mod -> sid -> k -> rec
    games = {}
    nskip = defaultdict(int)
    for f in glob.glob(os.path.join(RO, "*.json")):
        r = json.load(open(f))
        if r.get("skipped"):
            nskip[str(r["skipped"])] += 1
            continue
        raw[r["modality"]][r["state_id"]][r["k"]] = r
        games[r["state_id"]] = r.get("game_name") or r["state_id"]
    print(f"[F4] cells " + "  ".join(f"{m}={sum(len(v) for v in raw[m].values())}"
                                    for m in sorted(raw)) + f"  skips={dict(nskip)}")

    print(f"\n{'='*80}\nRAW DOSE CURVE (mean over states)\n{'='*80}")
    print(f"  {'modality':9s} {'k':>3s} {'n':>4s} {'logP(keyword)':>15s} "
          f"{'logP(control)':>15s} {'margin':>10s}")
    dose = {}
    for mod in sorted(raw):
        for k in KS:
            v = [r for s in raw[mod] for kk, r in raw[mod][s].items() if kk == k]
            if not v:
                continue
            kwv = [x["logp_keyword"] for x in v]
            cv = [x["logp_control"] for x in v if x.get("logp_control") is not None]
            mg = [x["margin"] for x in v if x.get("margin") is not None]
            print(f"  {mod:9s} {k:3d} {len(v):4d} {np.mean(kwv):15.4f} "
                  f"{(np.mean(cv) if cv else float('nan')):15.4f} "
                  f"{(np.mean(mg) if mg else float('nan')):10.4f}")
            dose[f"{mod}_k{k}"] = {"n": len(v), "logp_keyword": float(np.mean(kwv)),
                                   "logp_control": float(np.mean(cv)) if cv else None,
                                   "margin": float(np.mean(mg)) if mg else None}

    # per-state readout slopes
    sl = {}
    for mod in raw:
        sl[mod] = {}
        for sid, byk in raw[mod].items():
            kw = {k: r["logp_keyword"] for k, r in byk.items() if k in KS}
            mg = {k: r["margin"] for k, r in byk.items() if r.get("margin") is not None}
            if len(kw) == 3:
                sl[mod][sid] = {"kw_slope": fit(kw),
                                "margin_slope": fit(mg) if len(mg) >= 2 else None}

    print(f"\n{'='*80}\nREADOUT SLOPE, text vs visual (paired per state)\n{'='*80}")
    out = {"ks": KS, "dose": dose, "skips": dict(nskip)}
    if "text" in sl and "visual" in sl:
        common = sorted(set(sl["text"]) & set(sl["visual"]))
        print(f"  n paired states = {len(common)}")
        for field, label in (("kw_slope", "logP(keyword) slope, k={0,4,8}"),
                             ("margin_slope", "margin slope, k={4,8}")):
            t = [sl["text"][s][field] for s in common if sl["text"][s][field] is not None
                 and sl["visual"][s][field] is not None]
            v = [sl["visual"][s][field] for s in common if sl["text"][s][field] is not None
                 and sl["visual"][s][field] is not None]
            ok = [s for s in common if sl["text"][s][field] is not None
                  and sl["visual"][s][field] is not None]
            if not t:
                continue
            d = [a - b for a, b in zip(t, v)]
            mt, lot, hit = bootstrap_mean_ci(t)
            mv, lov, hiv = bootstrap_mean_ci(v)
            md, lod, hid = bootstrap_mean_ci(d)
            byg = defaultdict(list)
            for s, x in zip(ok, d):
                byg[games.get(s, s)].append(x)
            cm, clo, chi = clustered_ci(byg)
            npos, nneg, p = sign_test(d)
            print(f"\n  {label}")
            print(f"    text   {mt:8.4f} [{lot:7.4f},{hit:7.4f}]")
            print(f"    visual {mv:8.4f} [{lov:7.4f},{hiv:7.4f}]")
            print(f"    T-V    {md:8.4f} [{lod:7.4f},{hid:7.4f}]  "
                  f"game-clustered [{clo:7.4f},{chi:7.4f}]  {npos}+/{nneg}-  n={len(d)}")
            out[f"readout_{field}"] = {
                "n": len(d), "text": [mt, lot, hit], "visual": [mv, lov, hiv],
                "diff": [md, lod, hid], "diff_clustered": [cm, clo, chi],
                "n_pos": npos, "n_neg": nneg, "sign_p": p}
        print("\n  READING: T-V clearly > 0 => the keyword symbol becomes markedly")
        print("  more available on the language side when it arrives as text than as")
        print("  pixels -- the encoding difference, measured directly. T-V ~ 0 => the")
        print("  symbol is equally recoverable and the behavioural gap lies in how the")
        print("  action computation USES it, not in symbol availability.")

    # mediation
    act = action_slopes()
    if any(act.values()):
        print(f"\n{'='*80}\nMEDIATION (observational — correlation, not a causal claim)\n{'='*80}")
        med = {}
        for mod in sorted(sl):
            common = sorted(set(sl[mod]) & set(act.get(mod, {})))
            xs = [sl[mod][s]["kw_slope"] for s in common]
            ys = [act[mod][s] for s in common]
            if len(xs) < 10:
                continue
            r, _ = pearson(xs, ys)          # lib.stats returns (coefficient, n)
            rs, _ = spearman(xs, ys)
            print(f"  {mod:7s} n={len(xs):4d}  readout slope vs action slope: "
                  f"pearson r={r:+.3f}  spearman rho={rs:+.3f}")
            med[mod] = {"n": len(xs), "pearson": r, "spearman": rs}
        print("  Caveat: both slopes are computed from the same forward passes on the")
        print("  same documents, so shared state-level variance (prompt length, task")
        print("  family, base difficulty) inflates this. F1/F3 carry the causal weight.")
        out["mediation"] = med

    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\n[F4] wrote {OUT}")


if __name__ == "__main__":
    main()
