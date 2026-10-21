#!/usr/bin/env python3
"""
symbol-encoding / F0 — zero-GPU reanalysis of the EXISTING LI runs, split by how the
keyword tokenizes natively.

Free observational check on the same question F1/F2b answer experimentally: if
the text exposure effect rides on the keyword being ONE native discrete symbol,
then states whose keyword is already multi-token should show a WEAKER text slope,
while their visual slope should be unaffected (pixels have no BPE).

    single-token keywords : microwave, fridge, safe, cabinet, drawer
    multi-token  keywords : sinkbasin (3), desklamp (3)

*** CONFOUND — READ BEFORE BELIEVING ANY OF THIS ***
Keyword tokenization is almost perfectly confounded with TASK FAMILY here:
sinkbasin appears in clean tasks, desklamp in look-at-obj-in-light tasks. Those
task families differ in prompt length, candidate-action sets, and base difficulty.
So a group difference is NOT attributable to tokenization. This script exists to
(a) check the direction is at least consistent with F1 and (b) show per-keyword
slopes so a reader can see whether any split is driven by one keyword.
The decisive, within-state, task-matched version of this question is the casing
manipulation in run_text_casing.py (F1) and run_visual_casing.py (F2b).

Run: python3 reanalyze_li_by_tokenization.py
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
ANALYSIS_ROOT = os.path.dirname(MECH_ROOT)
sys.path.insert(0, MECH_ROOT)
from lib.stats import bootstrap_mean_ci, sign_test  # noqa: E402
from lib.config import MODEL_PATH  # noqa: E402

LI_RESULTS = os.path.join(ANALYSIS_ROOT, "lexical-interference", "results")
KS = [0, 4, 8]          # same k grid as symbol-encoding so the numbers are comparable
OUT = os.path.join(HERE, "results", "li_by_tokenization.json")


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


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    def ntok(w):
        return len(tok(" " + w, add_special_tokens=False)["input_ids"])

    # state -> rep -> k -> (logp_distractor, logp_correct)
    data = defaultdict(lambda: defaultdict(dict))
    meta = {}
    for f in glob.glob(os.path.join(LI_RESULTS, "*.json")):
        b = os.path.basename(f)
        if "__k-1__" in b:
            continue
        r = json.load(open(f))
        rep, k = r.get("representation"), r.get("level")
        if rep not in ("text", "visual") or k not in KS:
            continue
        cs = r.get("candidate_scores") or {}
        d, c = r.get("distractor_action"), r.get("correct_action")
        if d not in cs or c not in cs:
            continue
        sid = r["state_id"]
        data[sid][rep][k] = (cs[d]["action_logprob"], cs[c]["action_logprob"])
        meta[sid] = {"keyword": r["distractor_keyword"], "game_name": r.get("game_name"),
                     "task_type": r.get("task_type")}

    def slope(byk, idx):
        ks = sorted(byk)
        return float(np.polyfit(ks, [byk[k][idx] for k in ks], 1)[0])

    rows = []
    for sid, reps in data.items():
        if not all(len(reps.get(r, {})) == len(KS) for r in ("text", "visual")):
            continue
        kw = meta[sid]["keyword"]
        rows.append({
            "state_id": sid, "keyword": kw, "n_kw_tokens": ntok(kw),
            "game_name": meta[sid]["game_name"], "task_type": meta[sid]["task_type"],
            "slope_text": slope(reps["text"], 0), "slope_visual": slope(reps["visual"], 0),
            "slope_text_cor": slope(reps["text"], 1),
            "slope_visual_cor": slope(reps["visual"], 1),
        })
    print(f"[F0] {len(rows)} states with complete text+visual on k={KS}")
    if not rows:
        return

    print(f"\n{'='*84}")
    print("PER-KEYWORD (so you can see whether any split is one keyword's doing)")
    print(f"{'='*84}")
    print(f"{'keyword':12s} {'tok':>4s} {'n':>4s} {'slope_text':>11s} {'slope_vis':>11s} "
          f"{'T-V gap':>9s}  {'task types':<28s}")
    by_kw = defaultdict(list)
    for r in rows:
        by_kw[r["keyword"]].append(r)
    for kw in sorted(by_kw, key=lambda w: (ntok(w), w)):
        g = by_kw[kw]
        st = np.mean([x["slope_text"] for x in g])
        sv = np.mean([x["slope_visual"] for x in g])
        tt = ",".join(sorted({str(x["task_type"]) for x in g}))
        print(f"{kw:12s} {ntok(kw):4d} {len(g):4d} {st:11.4f} {sv:11.4f} "
              f"{st-sv:9.4f}  {tt[:28]:<28s}")

    print(f"\n{'='*84}")
    print("GROUPED BY TOKENIZATION  (confounded with task family — see docstring)")
    print(f"{'='*84}")
    groups = {"single_token": [r for r in rows if r["n_kw_tokens"] == 1],
              "multi_token": [r for r in rows if r["n_kw_tokens"] > 1]}
    out = {"n_states": len(rows), "ks": KS, "groups": {},
           "per_keyword": {kw: {"n_kw_tokens": ntok(kw), "n": len(g),
                                "slope_text": float(np.mean([x["slope_text"] for x in g])),
                                "slope_visual": float(np.mean([x["slope_visual"] for x in g])),
                                "task_types": sorted({str(x["task_type"]) for x in g})}
                           for kw, g in by_kw.items()},
           "CONFOUND": "keyword tokenization is confounded with task family "
                       "(sinkbasin<->clean, desklamp<->look-at-obj); suggestive only, "
                       "F1/F2b casing is the decisive within-state test"}
    for gname, g in groups.items():
        if not g:
            continue
        st = [x["slope_text"] for x in g]
        sv = [x["slope_visual"] for x in g]
        gap = [x["slope_text"] - x["slope_visual"] for x in g]
        byg = defaultdict(list)
        for x in g:
            byg[x["game_name"]].append(x["slope_text"] - x["slope_visual"])
        mt, lot, hit = bootstrap_mean_ci(st)
        mv, lov, hiv = bootstrap_mean_ci(sv)
        mg, log_, hig = bootstrap_mean_ci(gap)
        cm, clo, chi = clustered_ci(byg)
        npos, nneg, p = sign_test(gap)
        kws = sorted({x["keyword"] for x in g})
        print(f"\n{gname}  n={len(g)} states, {len(byg)} games, keywords={kws}")
        print(f"  slope text    {mt:8.4f} [{lot:7.4f},{hit:7.4f}]")
        print(f"  slope visual  {mv:8.4f} [{lov:7.4f},{hiv:7.4f}]")
        print(f"  T-V gap       {mg:8.4f} [{log_:7.4f},{hig:7.4f}]   "
              f"game-clustered [{clo:7.4f},{chi:7.4f}]  {npos}+/{nneg}-")
        print(f"  logP(correct) slopes: text {np.mean([x['slope_text_cor'] for x in g]):.4f}  "
              f"visual {np.mean([x['slope_visual_cor'] for x in g]):.4f}   (control: ~0)")
        out["groups"][gname] = {
            "n": len(g), "n_games": len(byg), "keywords": kws,
            "slope_text": [mt, lot, hit], "slope_visual": [mv, lov, hiv],
            "gap": [mg, log_, hig], "gap_clustered": [cm, clo, chi],
            "gap_n_pos": npos, "gap_n_neg": nneg, "gap_sign_p": p,
            "slope_text_cor": float(np.mean([x["slope_text_cor"] for x in g])),
            "slope_visual_cor": float(np.mean([x["slope_visual_cor"] for x in g]))}

    if len(out["groups"]) == 2:
        a, b = out["groups"]["single_token"], out["groups"]["multi_token"]
        print(f"\n{'='*84}\nBETWEEN-GROUP CONTRAST (unpaired, confounded — direction only)\n{'='*84}")
        print(f"  text slope   single-token {a['slope_text'][0]:.4f}  vs  "
              f"multi-token {b['slope_text'][0]:.4f}   diff {a['slope_text'][0]-b['slope_text'][0]:+.4f}")
        print(f"  visual slope single-token {a['slope_visual'][0]:.4f}  vs  "
              f"multi-token {b['slope_visual'][0]:.4f}   diff {a['slope_visual'][0]-b['slope_visual'][0]:+.4f}")
        print("  Consistent with the symbol account IFF the text diff is clearly")
        print("  positive AND larger than the visual diff. Confounded either way.")
        out["contrast"] = {
            "text_diff": a["slope_text"][0] - b["slope_text"][0],
            "visual_diff": a["slope_visual"][0] - b["slope_visual"][0]}

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\n[F0] wrote {OUT}")


if __name__ == "__main__":
    main()
