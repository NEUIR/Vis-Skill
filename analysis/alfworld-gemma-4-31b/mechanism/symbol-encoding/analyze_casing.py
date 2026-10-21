#!/usr/bin/env python3
"""
symbol-encoding analysis — the crossed casing design, effect sizes first.

DV: per-state exposure slope  d logP(distractor action) / dk  over k in {0,4,8},
fitted per (state, modality, arm). k=0 is the SHARED anchor: the k=0 document
contains no keyword, so every arm reuses the same k=0 cell. That is what makes
arms differ only in what the k=4 and k=8 keyword occurrences are made of.

Reported, in this order of importance:
  1. GATE — native slope vs the lexical-interference anchors (text 0.846 on {0,4,8}, visual
     0.552). If native is off, nothing downstream is interpretable.
  2. Per-arm mean slope + bootstrap CI, per modality.
  3. PAIRED delta-slope vs the native arm, per state, with bootstrap CI, sign
     test, and a game-clustered CI (165 states come from only 64 games, so the
     naive per-state CI is anticonservative).
  4. The INTERACTION: (native - altcase) in text  minus  (native - altcase) in
     visual, paired per state. This single number is the test.
  5. Slope regressed on the MEASURED keyword token count, not the nominal arm
     (`safe` stays 1 token even uppercased, `microwave` goes 1 -> 3 -> 6).
  6. Controls: logP(correct) slope per arm (must stay ~0 — casing must not
     manufacture a k-effect on the non-keyword action) and ctrl_altcase (text
     only; mangling a frequency-matched OTHER word must not move the slope).

Run: python3 analyze_casing.py
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
LI_ANCHOR = {"text": 0.846, "visual": 0.552}   # stem-only scoring on {0,4,8} / LI per-state
DIRS = {"text": os.path.join(HERE, "results", "text_casing"),
        "visual": os.path.join(HERE, "results", "visual_casing")}
OUT = os.path.join(HERE, "results", "casing_analysis.json")


def slope(byk: dict) -> float | None:
    ks = sorted(byk)
    if len(ks) < 2:
        return None
    return float(np.polyfit(ks, [byk[k] for k in ks], 1)[0])


def clustered_ci(vals_by_group: dict, n_boot: int = 4000, seed: int = 0):
    """Cluster bootstrap over games: resample GAMES, then average within game.

    165 states share 64 games; states from one game reuse the same skill corpus
    and task, so treating them as independent understates the CI.
    """
    groups = list(vals_by_group)
    if not groups:
        return None, None, None
    rng = np.random.default_rng(seed)
    means = [float(np.mean(vals_by_group[g])) for g in groups]
    point = float(np.mean(means))
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        boots.append(np.mean([means[i] for i in pick]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def load(modality: str):
    """-> cells[(sid, arm)][k] = rec, plus game map and skip tally."""
    cells = defaultdict(dict)
    game = {}
    ntok = defaultdict(list)
    skips = defaultdict(int)
    stem_bad = 0
    n = 0
    for f in glob.glob(os.path.join(DIRS[modality], "*.json")):
        r = json.load(open(f))
        if r.get("skipped"):
            skips[str(r["skipped"])] += 1
            continue
        if "logp_distractor" not in r:
            continue
        n += 1
        if not r.get("stem_prefix_ok", True):
            stem_bad += 1
        sid, k, arm = r["state_id"], r["k"], r["arm"]
        game[sid] = r.get("game_name") or sid
        cells[(sid, arm)][k] = r
        if k > 0 and modality == "text":
            ntok[arm].append(r.get("meta", {}).get("kw_token_count"))
    return cells, game, ntok, skips, stem_bad, n


def arms_present(cells):
    return sorted({arm for (_, arm) in cells if arm != "native"} | {"native"})


def slopes_for(cells, modality):
    """-> {arm: {sid: (slope_dis, slope_cor)}}, sharing the k=0 native anchor."""
    anchor = {sid: c for (sid, arm), c in cells.items() if arm == "native" and 0 in c}
    out = defaultdict(dict)
    for (sid, arm), byk in cells.items():
        dis = {k: r["logp_distractor"] for k, r in byk.items() if k in KS}
        cor = {k: r["logp_correct"] for k, r in byk.items() if k in KS}
        if 0 not in dis and sid in anchor and 0 in anchor[sid]:
            dis[0] = anchor[sid][0]["logp_distractor"]
            cor[0] = anchor[sid][0]["logp_correct"]
        sd, sc = slope(dis), slope(cor)
        if sd is not None and len(dis) == 3:
            out[arm][sid] = (sd, sc)
    return out


def report(modality, res, game):
    print(f"\n{'='*78}\n{modality.upper()}\n{'='*78}")
    arms = [a for a in ["native", "title", "upper", "altcase", "ctrl_altcase"] if a in res]
    print(f"{'arm':14s} {'n':>4s} {'slope_dis [95% CI]':>28s} "
          f"{'clustered CI (game)':>26s} {'slope_cor':>10s}")
    per_arm = {}
    for arm in arms:
        sids = sorted(res[arm])
        sd = [res[arm][s][0] for s in sids]
        sc = [res[arm][s][1] for s in sids if res[arm][s][1] is not None]
        m, lo, hi = bootstrap_mean_ci(sd)
        byg = defaultdict(list)
        for s in sids:
            byg[game.get(s, s)].append(res[arm][s][0])
        cm, clo, chi = clustered_ci(byg)
        print(f"{arm:14s} {len(sd):4d} {m:9.4f} [{lo:7.4f},{hi:7.4f}] "
              f"{cm:9.4f} [{clo:7.4f},{chi:7.4f}] {np.mean(sc):10.4f}")
        per_arm[arm] = {"n": len(sd), "slope_dis_mean": m, "ci": [lo, hi],
                        "clustered": [cm, clo, chi],
                        "slope_cor_mean": float(np.mean(sc)) if sc else None}
    if "native" in per_arm:
        a = LI_ANCHOR[modality]
        got = per_arm["native"]["slope_dis_mean"]
        ok = per_arm["native"]["ci"][0] <= a <= per_arm["native"]["ci"][1]
        print(f"\n  GATE native vs LI anchor {a:.3f}: got {got:.4f}  "
              f"{'PASS (anchor inside CI)' if ok else 'CHECK (anchor outside CI)'}")
        per_arm["_gate"] = {"anchor": a, "native": got, "anchor_in_ci": bool(ok)}

    print(f"\n  PAIRED delta-slope vs native (negative = casing WEAKENED exposure)")
    print(f"  {'arm':14s} {'n':>4s} {'delta [95% CI]':>26s} "
          f"{'clustered CI':>26s} {'sign':>12s} {'%of native':>10s}")
    paired = {}
    base = res.get("native", {})
    for arm in arms:
        if arm == "native":
            continue
        common = sorted(set(res[arm]) & set(base))
        if not common:
            continue
        d = [res[arm][s][0] - base[s][0] for s in common]
        m, lo, hi = bootstrap_mean_ci(d)
        byg = defaultdict(list)
        for s in common:
            byg[game.get(s, s)].append(res[arm][s][0] - base[s][0])
        cm, clo, chi = clustered_ci(byg)
        npos, nneg, p = sign_test(d)
        nb = per_arm["native"]["slope_dis_mean"]
        print(f"  {arm:14s} {len(d):4d} {m:8.4f} [{lo:7.4f},{hi:7.4f}] "
              f"{cm:8.4f} [{clo:7.4f},{chi:7.4f}] {npos:4d}+/{nneg:4d}- "
              f"{100*m/nb if nb else float('nan'):9.1f}%")
        paired[arm] = {"n": len(d), "delta_mean": m, "ci": [lo, hi],
                       "clustered": [cm, clo, chi], "n_pos": npos, "n_neg": nneg,
                       "sign_p": p, "pct_of_native": 100 * m / nb if nb else None}
    return {"per_arm": per_arm, "paired_vs_native": paired,
            "slopes": {a: {s: v[0] for s, v in res[a].items()} for a in res}}


def main():
    out = {"ks": KS, "li_anchor": LI_ANCHOR, "modalities": {}}
    slopes_by_mod = {}
    games = {}
    for mod in ("text", "visual"):
        if not os.path.isdir(DIRS[mod]):
            print(f"[{mod}] no results dir yet — skipping")
            continue
        cells, game, ntok, skips, stem_bad, n = load(mod)
        if not cells:
            print(f"[{mod}] no cells yet — skipping")
            continue
        games.update(game)
        print(f"[{mod}] cells={n} states={len({s for s,_ in cells})} "
              f"stem_prefix_bad={stem_bad} skips={dict(skips)}")
        if mod == "text":
            for arm, v in sorted(ntok.items()):
                v = [x for x in v if x is not None]
                if v:
                    print(f"    measured kw tokens  {arm:14s} mean={np.mean(v):.2f} "
                          f"range={min(v)}-{max(v)}")
        res = slopes_for(cells, mod)
        slopes_by_mod[mod] = res
        out["modalities"][mod] = report(mod, res, game)
        out["modalities"][mod]["n_cells"] = n
        out["modalities"][mod]["stem_prefix_bad"] = stem_bad
        out["modalities"][mod]["skips"] = dict(skips)

    # ── THE INTERACTION ────────────────────────────────────────────────
    if "text" in slopes_by_mod and "visual" in slopes_by_mod:
        T, V = slopes_by_mod["text"], slopes_by_mod["visual"]
        print(f"\n{'='*78}\nMODALITY x CASING INTERACTION  (the decisive test)\n{'='*78}")
        print("  d_text = slope(text,native) - slope(text,ARM)   [how much casing cost TEXT]")
        print("  d_vis  = slope(vis, native) - slope(vis, ARM)   [how much it cost VISUAL]")
        print("  H predicts d_text > 0 and d_vis ~ 0, so the interaction is POSITIVE.\n")
        print(f"  {'arm':10s} {'n':>4s} {'d_text':>18s} {'d_visual':>18s} "
              f"{'interaction [95% CI]':>26s} {'sign':>11s}")
        inter = {}
        for arm in ("upper", "altcase"):
            if arm not in T or arm not in V:
                continue
            common = sorted(set(T[arm]) & set(V[arm]) & set(T["native"]) & set(V["native"]))
            if not common:
                continue
            dt = [T["native"][s][0] - T[arm][s][0] for s in common]
            dv = [V["native"][s][0] - V[arm][s][0] for s in common]
            ix = [a - b for a, b in zip(dt, dv)]
            mt, lot, hit = bootstrap_mean_ci(dt)
            mv, lov, hiv = bootstrap_mean_ci(dv)
            mi, loi, hii = bootstrap_mean_ci(ix)
            byg = defaultdict(list)
            for s, v in zip(common, ix):
                byg[games.get(s, s)].append(v)
            cm, clo, chi = clustered_ci(byg)
            npos, nneg, p = sign_test(ix)
            print(f"  {arm:10s} {len(ix):4d} {mt:7.4f}[{lot:6.3f},{hit:6.3f}] "
                  f"{mv:7.4f}[{lov:6.3f},{hiv:6.3f}] "
                  f"{mi:8.4f} [{loi:7.4f},{hii:7.4f}] {npos:4d}+/{nneg:4d}-")
            print(f"  {'':10s}      game-clustered interaction CI: "
                  f"{cm:.4f} [{clo:.4f},{chi:.4f}]")
            inter[arm] = {"n": len(ix), "d_text": [mt, lot, hit],
                          "d_visual": [mv, lov, hiv],
                          "interaction": [mi, loi, hii],
                          "interaction_clustered": [cm, clo, chi],
                          "n_pos": npos, "n_neg": nneg, "sign_p": p}
        out["interaction"] = inter

        # ── STEP vs GRADED: the dose contrast within each modality ──────
        # upper and altcase both break exact symbol identity, but altcase is far
        # more visually irregular AND has ~2x the BPE tokens. So:
        #   discrete-symbol matching  -> once identity is broken the cost is
        #       SATURATED: (altcase - upper) ~ 0.  A step function.
        #   perceptual re-recognition -> the cost keeps GROWING with irregularity:
        #       (altcase - upper) < 0.  A graded function.
        # Differencing within modality first cancels the large baseline level gap
        # between text and visual, which the raw interaction above does not.
        print(f"\n{'='*78}\nSTEP vs GRADED: does the cost keep growing with mangling?\n{'='*78}")
        grad = {}
        for mod, R in (("text", T), ("visual", V)):
            if "upper" not in R or "altcase" not in R:
                continue
            cs = sorted(set(R["upper"]) & set(R["altcase"]))
            d = [R["altcase"][s][0] - R["upper"][s][0] for s in cs]
            m, lo, hi = bootstrap_mean_ci(d)
            npos, nneg, p = sign_test(d)
            byg_m = defaultdict(list)
            for s, x in zip(cs, d):
                byg_m[games.get(s, s)].append(x)
            gm, glo, ghi = clustered_ci(byg_m)
            # Verdict on the GAME-CLUSTERED CI, the primary interval here. Note a
            # small POSITIVE value is still "not graded" -- it means extra mangling
            # bought no extra cost. Only a clearly negative value is graded decay.
            if ghi < 0:
                verdict = "GRADED (cost grows with mangling)"
            elif glo <= 0 <= ghi:
                verdict = "SATURATED (step; no extra cost)"
            else:
                verdict = "NOT GRADED (slightly positive; no extra cost)"
            print(f"  {mod:7s} altcase-upper {m:+.4f} [{lo:+.4f},{hi:+.4f}]  "
                  f"clustered [{glo:+.4f},{ghi:+.4f}]  {npos}+/{nneg}- p={p:.3g}  "
                  f"n={len(d)}\n  {'':9s}-> {verdict}")
            grad[mod] = {"n": len(d), "delta": [m, lo, hi],
                         "clustered": [gm, glo, ghi], "n_pos": npos,
                         "n_neg": nneg, "sign_p": p, "verdict": verdict}
        if len(grad) == 2:
            cs = sorted(set(T["upper"]) & set(T["altcase"])
                        & set(V["upper"]) & set(V["altcase"]))
            dt = [T["altcase"][s][0] - T["upper"][s][0] for s in cs]
            dv = [V["altcase"][s][0] - V["upper"][s][0] for s in cs]
            ix = [a - b for a, b in zip(dt, dv)]
            mi, loi, hii = bootstrap_mean_ci(ix)
            byg = defaultdict(list)
            for s, x in zip(cs, ix):
                byg[games.get(s, s)].append(x)
            cm, clo, chi = clustered_ci(byg)
            npos, nneg, p = sign_test(ix)
            print(f"\n  DOSE INTERACTION  (text altcase-upper) - (visual altcase-upper)")
            print(f"    {mi:+.4f} [{loi:+.4f},{hii:+.4f}]  "
                  f"game-clustered [{clo:+.4f},{chi:+.4f}]  {npos}+/{nneg}-  n={len(ix)}")
            print(f"    > 0 means text saturates while visual keeps degrading: the text")
            print(f"    keyword is matched as a DISCRETE SYMBOL (all-or-nothing), the")
            print(f"    visual keyword is RE-RECOGNISED perceptually (degrades by degree).")
            grad["dose_interaction"] = {"n": len(ix), "value": [mi, loi, hii],
                                        "clustered": [cm, clo, chi], "n_pos": npos,
                                        "n_neg": nneg, "sign_p": p}
        out["step_vs_graded"] = grad

        print("\n  READING THE RESULT")
        print("   d_text>0, d_vis~0, interaction CI excludes 0"
              "  -> text rides on the keyword being a NATIVE DISCRETE TOKEN;")
        print("      the modality gap is an ENCODING difference. (hypothesis supported)")
        print("   d_text~d_vis>0 (interaction ~0)"
              "        -> surface-form/salience, NOT segmentation.")
        print("   d_text~0                          "
              "        -> exposure does not need a crisp symbol; look elsewhere.")

    # ── COUNT, isolated: upper vs altcase with identity broken in BOTH ──
    # Both arms mismatch the lowercase scored action, so symbol-identity breakage
    # is held constant; only the token count differs (e.g. microwave 3 vs 6).
    # This is the cleanest count test available, and unlike the pooled regression
    # it is paired within state and within keyword.
    if "text" in slopes_by_mod:
        ntok_cell = {}
        for f in glob.glob(os.path.join(DIRS["text"], "*.json")):
            r = json.load(open(f))
            if r.get("skipped") or r["k"] == 0:
                continue
            n = r.get("meta", {}).get("kw_token_count")
            if n:
                ntok_cell[(r["state_id"], r["arm"])] = n
        T = slopes_by_mod["text"]
        if "upper" in T and "altcase" in T:
            pairs = [s for s in sorted(set(T["upper"]) & set(T["altcase"]))
                     if ntok_cell.get((s, "altcase"), 0) > ntok_cell.get((s, "upper"), 0)]
            if pairs:
                d = [T["altcase"][s][0] - T["upper"][s][0] for s in pairs]
                m, lo, hi = bootstrap_mean_ci(d)
                byg = defaultdict(list)
                for s, x in zip(pairs, d):
                    byg[games.get(s, s)].append(x)
                cm, clo, chi = clustered_ci(byg)
                npos, nneg, p = sign_test(d)
                extra = float(np.mean([ntok_cell[(s, "altcase")] - ntok_cell[(s, "upper")]
                                       for s in pairs]))
                print(f"\n{'='*78}\nCOUNT ISOLATED: altcase - upper, identity broken in both\n{'='*78}")
                print(f"  n={len(d)} states where altcase has MORE keyword tokens than upper "
                      f"(mean +{extra:.1f} tokens)")
                print(f"  delta slope {m:+.4f} [{lo:+.4f},{hi:+.4f}]  "
                      f"game-clustered [{clo:+.4f},{chi:+.4f}]  {npos}+/{nneg}-  p={p:.3g}")
                print(f"  CI containing 0 => extra fragmentation costs nothing once the")
                print(f"  symbol already fails to match: the carrier is IDENTITY, not COUNT.")
                out["count_isolated_altcase_minus_upper"] = {
                    "n": len(d), "mean_extra_tokens": extra, "delta": [m, lo, hi],
                    "clustered": [cm, clo, chi], "n_pos": npos, "n_neg": nneg,
                    "sign_p": p}

    # ── slope vs MEASURED token count (text) ──────────────────────────
    if "text" in slopes_by_mod:
        rows = []
        for f in glob.glob(os.path.join(DIRS["text"], "*.json")):
            r = json.load(open(f))
            if r.get("skipped") or r["k"] == 0 or r["arm"] == "ctrl_altcase":
                continue
            n = r.get("meta", {}).get("kw_token_count")
            s = slopes_by_mod["text"].get(r["arm"], {}).get(r["state_id"])
            if n and s:
                rows.append((n, s[0]))
        if len(rows) > 10:
            n_arr = np.array([r[0] for r in rows], float)
            s_arr = np.array([r[1] for r in rows], float)
            b, a = np.polyfit(n_arr, s_arr, 1)
            print(f"\n  TEXT slope vs MEASURED keyword token count: "
                  f"slope={b:+.4f} per extra token, intercept={a:.4f}, n={len(rows)}")
            print(f"    per-token-count means: " + "  ".join(
                f"{int(t)}tok:{s_arr[n_arr==t].mean():.3f}(n={int((n_arr==t).sum())})"
                for t in sorted(set(n_arr))))
            out["text_slope_vs_token_count"] = {
                "regression_slope": float(b), "intercept": float(a), "n": len(rows),
                "by_count": {int(t): [float(s_arr[n_arr == t].mean()),
                                      int((n_arr == t).sum())] for t in sorted(set(n_arr))}}

    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\n[analyze] wrote {OUT}")


if __name__ == "__main__":
    main()
