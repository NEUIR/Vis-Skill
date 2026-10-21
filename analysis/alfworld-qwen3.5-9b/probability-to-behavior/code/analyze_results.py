"""
Step 4 — analyse the probability-to-behavior bridge.

The chain this experiment closes:

    Skill keyword frequency k ↑  →  probability of the wrong competitor ↑  →  the model actually executes the wrong action
                      (forced decoding)                   (free generation —
                                                           and the competitor IS
                                                           executable, so this is
                                                           real wrong behaviour)

and, with the optional rollout stage:

    →  real end-to-end task cost (extra steps / task failure).

Inputs: probe cells `{state_id}__k{k}__{rep}.json` (run_probe.py) and, if
present, rollout cells `rollout_results/*__rollout_*.json` (run_rollout.py).

Analyses produced
=================

1. **(level × representation) aggregation** — P(competitor), logP(competitor),
   margin, entropy, ranks + the behavioural rates that matter here:
   chose_competitor_rate (free generation actually picked the executable
   wrong action), error_rate, repro rates. The no-skill control (level -1)
   anchors every curve.

2. **Trends vs k** per representation (aggregate OLS + per-state paired
   slopes with exact sign tests), including the modality/structure
   decomposition carried over from ../rasterized-interference:
       total = slope(text) − slope(structural)
       modality = slope(text) − slope(rasterized)
       structure = slope(rasterized) − slope(structural)
   The bridge claim: chose_competitor_rate has a POSITIVE trend in text, and a
   flatter one in the image conditions.

3. **Probability-to-behavior consistency** — the actual bridge statistics:
     * mean P(competitor) in cells where free generation chose it vs cells
       where it did not, per representation (+ point-biserial correlation
       between p_distractor_sum and the binary choice over all cells);
     * contingency: how often the forced-decoding argmax IS the competitor vs
       how often generation picks it;
     * per-state paired k0→kmax deltas of probability AND behaviour, and the
       correlation between Δp and Δchoice across states (states whose
       probability moved most should be the ones whose behaviour moved).

4. **Rollout summary (optional)** — per (rep, k): success rate within budget
   and mean policy steps for the correct vs the competitor branch; on pairs
   where both succeed, the extra steps the competitor cost; branch-level
   "Nothing happened" rates. This converts the single-step probability shift
   into end-to-end task cost.

5. **QA** — incomplete decodings, unparsable generations, competitor
   admissibility re-checked against the states, rollout bookkeeping.

Outputs (under --output_dir):
  probability_to_behavior_analysis.json
  per_trial.csv / by_level_rep.csv / bridge_states.csv
"""
import os
import re
import csv
import json
import math
import argparse
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))
import skill_lexicon as lex  # noqa: E402

RESULT_GLOB = "*__k*__*.json"
ROLLOUT_GLOB = "*__rollout_*.json"

REP_ALIAS = {"visual": "structural"}
MAIN_REPS = ("text", "rasterized", "structural")

DECOMPOSITION = (
    ("text", "structural", "total"),
    ("text", "rasterized", "modality"),
    ("rasterized", "structural", "structure"),
)

DIST_METRICS = [
    "p_correct_sum", "p_distractor_sum",
    "p_correct_mean", "p_distractor_mean",
    "entropy_sum", "entropy_norm_sum",
    "entropy_mean", "entropy_norm_mean",
    "logp_correct_action", "logp_distractor_action", "logp_margin_action",
    "logp_correct", "logp_distractor", "logp_margin",
    "rank_correct_sum", "rank_distractor_sum",
]

PER_STATE_TREND_METRICS = [
    "logp_distractor_action", "logp_correct_action", "logp_margin_action",
    "entropy_sum", "p_distractor_sum", "p_correct_sum",
]

TREND_METRICS = [
    "logp_distractor_action", "logp_correct_action", "logp_margin_action",
    "entropy_sum", "p_distractor_sum", "p_correct_sum",
    "error_rate", "chose_distractor_rate", "repro_keyword_rate",
    "argmax_distractor_rate",
]

DELTA_METRICS = ["logp_distractor_action", "logp_correct_action",
                 "entropy_sum", "p_distractor_sum"]


# ══════════════════════════════════════════════════════════════════════
# Loading / flattening
# ══════════════════════════════════════════════════════════════════════

def load_trials(results_dirs):
    trials, bad_files = [], []
    source_counts = defaultdict(int)
    seen = set()
    for d in results_dirs:
        d = Path(d)
        if not d.exists():
            print(f"[WARN] results dir not found, skipping: {d}")
            continue
        for f in sorted(d.glob(RESULT_GLOB)):
            if "__rollout_" in f.name:
                continue
            try:
                rec = json.loads(f.read_text(encoding="utf-8"))
            except Exception as e:
                bad_files.append((str(f), str(e)))
                continue
            rep = REP_ALIAS.get(rec.get("representation"), rec.get("representation"))
            key = (rec.get("state_id"), rec.get("level"), rep)
            if key in seen:
                continue
            seen.add(key)
            rec["representation"] = rep
            rec["_file"] = f.name
            rec["_source_dir"] = str(d)
            source_counts[str(d)] += 1
            trials.append(rec)
    return trials, bad_files, dict(source_counts)


def load_states(states_dir):
    out = {}
    states_dir = Path(states_dir)
    if not states_dir.exists():
        return out
    for f in states_dir.glob("*.json"):
        if f.name == "manifest.json":
            continue
        try:
            s = json.loads(f.read_text(encoding="utf-8"))
            out[s["state_id"]] = s
        except Exception:
            continue
    return out


def load_documents(variants_dir):
    docs = {}
    text_dir = Path(variants_dir) / "text" if variants_dir else None
    if not text_dir or not text_dir.exists():
        return docs
    for state_dir in text_dir.iterdir():
        if not state_dir.is_dir():
            continue
        for md in state_dir.glob("level_*.md"):
            m = re.match(r"level_(-?\d+)\.md", md.name)
            if m:
                docs[(state_dir.name, int(m.group(1)))] = md.read_text(encoding="utf-8")
    return docs


def load_rollouts(rollout_dir):
    out = []
    d = Path(rollout_dir) if rollout_dir else None
    if not d or not d.exists():
        return out
    for f in sorted(d.glob(ROLLOUT_GLOB)):
        try:
            out.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception:
            continue
    return out


def context_words(state):
    if not state:
        return set()
    text = [state.get("query", ""), state.get("correct_action", "")]
    for msg in state.get("history_messages", []):
        c = msg.get("content", "")
        if isinstance(c, str):
            text.append(c)
    return set(lex.content_words(" ".join(text)))


def flatten_trial(rec, state, doc_text):
    dist = rec.get("distribution") or {}
    gen = rec.get("generation") or {}

    chosen = gen.get("chosen_action", "") or ""
    keyword = rec.get("distractor_keyword", "")
    phrase = rec.get("distractor_phrase", "")

    chose_correct = bool(gen.get("chose_correct"))
    unparsable = not chosen
    is_error = (not chose_correct) and not unparsable

    verb_surface = lex.VERB_SURFACE.get(rec.get("distractor_action_type", ""),
                                        rec.get("distractor_action_type", "") or "")
    repro_keyword = is_error and lex.count_word(chosen, keyword) > 0
    repro_exact = is_error and bool(gen.get("chose_distractor"))
    repro_phrase = is_error and (
        lex.count_phrase(chosen, phrase) > 0
        or lex.count_verb_noun(chosen, verb_surface, keyword) > 0)
    repro_other = False
    if is_error and doc_text:
        doc_only = set(lex.content_words(doc_text)) - context_words(state)
        doc_only -= {keyword.lower()} | set(lex.content_words(keyword))
        doc_only -= set(lex.content_words(phrase))
        repro_other = any(w in doc_only for w in lex.content_words(chosen))

    row = {
        "state_id": rec.get("state_id"),
        "level": rec.get("level"),
        "representation": rec.get("representation"),
        "source_dir": rec.get("_source_dir"),
        "task_type": rec.get("task_type"),
        "step_idx": rec.get("step_idx"),
        "correct_action": rec.get("correct_action"),
        "distractor_action": rec.get("distractor_action"),
        "distractor_keyword": keyword,
        "competitor_admissible": bool(rec.get("competitor_admissible",
                                              (state or {}).get("competitor_admissible"))),
        "chosen_action": chosen,
        "chosen_verb": gen.get("chosen_verb", ""),
        "chose_correct": chose_correct,
        "chose_distractor": bool(gen.get("chose_distractor")),
        "chosen_in_candidates": bool(gen.get("chosen_in_candidates")),
        "unparsable": unparsable,
        "is_error": is_error,
        "repro_keyword": repro_keyword,
        "repro_exact": repro_exact,
        "repro_phrase": repro_phrase,
        "repro_other_skillword": repro_other,
        "argmax_sum": dist.get("argmax_sum", ""),
        "argmax_is_correct": dist.get("argmax_sum") == rec.get("correct_action"),
        "argmax_is_distractor": dist.get("argmax_sum") == rec.get("distractor_action"),
        "n_scored": dist.get("n_scored", 0),
        "n_candidates": dist.get("n_candidates", 0),
        "fully_scored": bool(dist.get("n_scored")) and
                        dist.get("n_scored") == dist.get("n_candidates"),
        "stem_prefix_ok": bool(dist.get("stem_prefix_ok", True)),
        "response_excerpt": (gen.get("response_text", "") or "")[:240],
    }
    for m in DIST_METRICS:
        row[m] = dist.get(m)
    return row


# ══════════════════════════════════════════════════════════════════════
# Stats helpers
# ══════════════════════════════════════════════════════════════════════

def _mean_sem(vals):
    vals = [v for v in vals if v is not None]
    n = len(vals)
    if n == 0:
        return {"mean": None, "sem": None, "std": None, "n": 0}
    mean = sum(vals) / n
    if n > 1:
        std = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1))
        sem = std / math.sqrt(n)
    else:
        std, sem = 0.0, 0.0
    return {"mean": mean, "sem": sem, "std": std, "n": n}


def _rate(num, den):
    return (num / den) if den else None


def linear_trend(xs, ys):
    if len(xs) < 2:
        return {"slope": None, "intercept": None, "r_squared": None, "n_points": len(xs)}
    n = len(xs)
    xm, ym = sum(xs) / n, sum(ys) / n
    ssxx = sum((x - xm) ** 2 for x in xs)
    ssxy = sum((x - xm) * (y - ym) for x, y in zip(xs, ys))
    ssyy = sum((y - ym) ** 2 for y in ys)
    if ssxx == 0:
        return {"slope": 0.0, "intercept": ym, "r_squared": 0.0, "n_points": n}
    slope = ssxy / ssxx
    return {"slope": slope, "intercept": ym - slope * xm,
            "r_squared": (ssxy ** 2) / (ssxx * ssyy) if ssyy > 0 else 0.0,
            "n_points": n}


def pearson(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 3:
        return {"r": None, "n": len(pairs)}
    xs2, ys2 = zip(*pairs)
    n = len(pairs)
    xm, ym = sum(xs2) / n, sum(ys2) / n
    sxy = sum((x - xm) * (y - ym) for x, y in pairs)
    sxx = sum((x - xm) ** 2 for x in xs2)
    syy = sum((y - ym) ** 2 for y in ys2)
    if sxx <= 0 or syy <= 0:
        return {"r": None, "n": n}
    return {"r": sxy / math.sqrt(sxx * syy), "n": n}


def sign_test_p(diffs):
    nz = [d for d in diffs if d is not None and d != 0]
    n = len(nz)
    if n == 0:
        return None
    k = sum(1 for d in nz if d > 0)
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) * (0.5 ** n)
    return min(1.0, 2 * tail)


def _diff_summary(diffs):
    return {**_mean_sem(diffs),
            "sign_test_p": sign_test_p(diffs),
            "n_positive": sum(1 for d in diffs if d is not None and d > 0),
            "n_negative": sum(1 for d in diffs if d is not None and d < 0)}


def aggregate(rows):
    n = len(rows)
    counts = {
        "n_unparsable": sum(r["unparsable"] for r in rows),
        "n_error": sum(not r["chose_correct"] for r in rows),
        "n_parsable_error": sum(r["is_error"] for r in rows),
        "n_chose_distractor": sum(r["chose_distractor"] for r in rows),
        "n_repro_keyword": sum(r["repro_keyword"] for r in rows),
        "n_repro_exact": sum(r["repro_exact"] for r in rows),
        "n_argmax_correct": sum(r["argmax_is_correct"] for r in rows),
        "n_argmax_distractor": sum(r["argmax_is_distractor"] for r in rows),
        "n_fully_scored": sum(r["fully_scored"] for r in rows),
        "n_stem_prefix_ok": sum(r["stem_prefix_ok"] for r in rows),
    }
    pe = counts["n_parsable_error"]
    rates = {
        "correct_rate": _rate(n - counts["n_error"], n),
        "error_rate": _rate(counts["n_error"], n),
        "unparsable_rate": _rate(counts["n_unparsable"], n),
        # THE behavioural metric of this experiment: the free generation picked
        # the executable-but-wrong competitor.
        "chose_distractor_rate": _rate(counts["n_chose_distractor"], n),
        "argmax_correct_rate": _rate(counts["n_argmax_correct"], n),
        "argmax_distractor_rate": _rate(counts["n_argmax_distractor"], n),
        "fully_scored_rate": _rate(counts["n_fully_scored"], n),
        "repro_keyword_rate": _rate(counts["n_repro_keyword"], pe),
        "repro_exact_rate": _rate(counts["n_repro_exact"], pe),
    }
    means = {m: _mean_sem([r[m] for r in rows]) for m in DIST_METRICS}
    return {"n": n, "counts": counts, "rates": rates, "means": means}


def metric_value(agg, metric):
    if metric in agg["means"]:
        return agg["means"][metric]["mean"]
    if metric in agg["rates"]:
        return agg["rates"][metric]
    return None


# ══════════════════════════════════════════════════════════════════════
# Trends (aggregate + per-state + decomposition)
# ══════════════════════════════════════════════════════════════════════

def compute_trends(by_level_rep, levels):
    trends = {}
    for rep in MAIN_REPS:
        per_metric = {}
        for metric in TREND_METRICS:
            xs, ys = [], []
            for k in levels:
                agg = by_level_rep.get(str(k), {}).get(rep)
                if not agg:
                    continue
                v = metric_value(agg, metric)
                if v is not None:
                    xs.append(float(k))
                    ys.append(float(v))
            per_metric[metric] = linear_trend(xs, ys)
        if any(t["slope"] is not None for t in per_metric.values()):
            trends[rep] = per_metric

    decomposition = {}
    for metric in TREND_METRICS:
        slopes = {rep: trends.get(rep, {}).get(metric, {}).get("slope")
                  for rep in MAIN_REPS}
        entry = {"slopes": slopes}
        for a, b, name in DECOMPOSITION:
            sa, sb = slopes.get(a), slopes.get(b)
            entry[name] = (sa - sb) if (sa is not None and sb is not None) else None
        tot = entry.get("total")
        entry["modality_share"] = (entry["modality"] / tot) if (
            tot not in (None, 0) and entry.get("modality") is not None) else None
        entry["structure_share"] = (entry["structure"] / tot) if (
            tot not in (None, 0) and entry.get("structure") is not None) else None
        decomposition[metric] = entry
    trends["slope_decomposition"] = decomposition
    return trends


def compute_per_state_trends(rows, levels):
    if len(levels) < 2:
        return {}
    idx = {}
    for r in rows:
        if r["representation"] in MAIN_REPS:
            idx[(r["state_id"], r["level"], r["representation"])] = r
    state_ids = sorted({sid for sid, _, _ in idx})

    out = {"levels": levels, "n_states": len(state_ids), "metrics": {}}
    for metric in PER_STATE_TREND_METRICS:
        slopes = {rep: {} for rep in MAIN_REPS}
        for rep in MAIN_REPS:
            for sid in state_ids:
                xs, ys = [], []
                for k in levels:
                    r = idx.get((sid, k, rep))
                    if r is None or r.get(metric) is None:
                        continue
                    xs.append(float(k))
                    ys.append(float(r[metric]))
                if len(xs) >= 2:
                    s = linear_trend(xs, ys)["slope"]
                    if s is not None:
                        slopes[rep][sid] = s
        entry = {rep: _mean_sem(list(slopes[rep].values())) for rep in MAIN_REPS}
        entry["n_states_per_rep"] = {rep: len(slopes[rep]) for rep in MAIN_REPS}
        for a, b, name in DECOMPOSITION:
            diffs = [slopes[a][sid] - slopes[b][sid]
                     for sid in sorted(slopes[a]) if sid in slopes[b]]
            entry[name] = _diff_summary(diffs)
        out["metrics"][metric] = entry
    return out


def compute_paired_deltas(rows, levels):
    if not levels or max(levels) == 0:
        return {}
    kmax = max(levels)
    idx = {(r["state_id"], r["level"], r["representation"]): r for r in rows}
    state_ids = sorted({r["state_id"] for r in rows
                        if r["representation"] in MAIN_REPS})

    out = {"k_max": kmax, "n_states": len(state_ids), "metrics": {}}
    for metric in DELTA_METRICS:
        per_rep = {}
        for rep in MAIN_REPS:
            vals = {}
            for sid in state_ids:
                r0 = idx.get((sid, 0, rep))
                r8 = idx.get((sid, kmax, rep))
                if r0 and r8 and r0.get(metric) is not None and r8.get(metric) is not None:
                    vals[sid] = r8[metric] - r0[metric]
            per_rep[rep] = vals
        entry = {rep: _mean_sem(list(per_rep[rep].values())) for rep in MAIN_REPS}
        entry["n_states_per_rep"] = {rep: len(per_rep[rep]) for rep in MAIN_REPS}
        for a, b, name in DECOMPOSITION:
            diffs = [per_rep[a][sid] - per_rep[b][sid]
                     for sid in sorted(per_rep[a]) if sid in per_rep[b]]
            entry[name] = _diff_summary(diffs)
        out["metrics"][metric] = entry
    return out


# ══════════════════════════════════════════════════════════════════════
# The bridge itself
# ══════════════════════════════════════════════════════════════════════

def compute_bridge(rows, levels):
    """Probability-to-behavior consistency, per representation.

    * mean P(competitor) among cells that chose it vs cells that did not
      (if behaviour follows probability, the chose-cells sit far higher);
    * point-biserial correlation between p_distractor_sum and the binary
      choice over all parsable cells;
    * argmax contingency: P(free generation chose competitor | forced argmax
      is the competitor) vs P(chose | argmax is not);
    * per-state paired k0→kmax: Δp_distractor_sum vs Δchose (0/1), correlated
      across states — did the states whose probability moved also move in
      behaviour?
    """
    out = {}
    idx = {(r["state_id"], r["level"], r["representation"]): r for r in rows}
    for rep in MAIN_REPS:
        rep_rows = [r for r in rows if r["representation"] == rep
                    and r["level"] is not None and r["level"] >= 0
                    and not r["unparsable"]]
        if not rep_rows:
            continue
        chose = [r for r in rep_rows if r["chose_distractor"]]
        not_chose = [r for r in rep_rows if not r["chose_distractor"]]
        pb = pearson([r["p_distractor_sum"] for r in rep_rows],
                     [1.0 if r["chose_distractor"] else 0.0 for r in rep_rows])
        argmax_d = [r for r in rep_rows if r["argmax_is_distractor"]]
        argmax_nd = [r for r in rep_rows if not r["argmax_is_distractor"]]
        entry = {
            "n_cells": len(rep_rows),
            "p_distractor_when_chose": _mean_sem([r["p_distractor_sum"] for r in chose]),
            "p_distractor_when_not_chose": _mean_sem([r["p_distractor_sum"] for r in not_chose]),
            "logp_distractor_when_chose": _mean_sem([r["logp_distractor_action"] for r in chose]),
            "logp_distractor_when_not_chose": _mean_sem(
                [r["logp_distractor_action"] for r in not_chose]),
            "point_biserial_p_vs_choice": pb,
            "chose_rate_given_argmax_distractor": _rate(
                sum(r["chose_distractor"] for r in argmax_d), len(argmax_d)),
            "chose_rate_given_argmax_not_distractor": _rate(
                sum(r["chose_distractor"] for r in argmax_nd), len(argmax_nd)),
            "n_argmax_distractor": len(argmax_d),
        }
        # per-state paired k0 → kmax probability vs behaviour movement
        if levels and max(levels) > 0:
            kmax = max(levels)
            dp, dc = [], []
            for sid in sorted({r["state_id"] for r in rep_rows}):
                r0 = idx.get((sid, 0, rep))
                r8 = idx.get((sid, kmax, rep))
                if (r0 and r8 and r0.get("p_distractor_sum") is not None
                        and r8.get("p_distractor_sum") is not None):
                    dp.append(r8["p_distractor_sum"] - r0["p_distractor_sum"])
                    dc.append(float(r8["chose_distractor"]) - float(r0["chose_distractor"]))
            entry["delta_p_k0_kmax"] = _mean_sem(dp)
            entry["delta_chose_k0_kmax"] = _mean_sem(dc)
            entry["corr_delta_p_delta_chose"] = pearson(dp, dc)
            entry["n_states_both_ended"] = len(dp)
        out[rep] = entry
    return out


def compute_rollout_summary(rollouts, levels):
    """Optional end-to-end cost of executing the competitor."""
    if not rollouts:
        return None
    by_group = defaultdict(list)
    for r in rollouts:
        if not r.get("ok", True) and "success" not in r:
            continue
        by_group[(r.get("representation"), r.get("level"), r.get("branch"))].append(r)

    def summarise(rs):
        n = len(rs)
        succ = [r for r in rs if r.get("success")]
        return {
            "n": n,
            "success_rate": _rate(len(succ), n),
            "mean_policy_steps": _mean_sem([r.get("policy_steps_after_branch") for r in rs]),
            "mean_nothing_happened": _mean_sem([r.get("nothing_happened_steps") for r in rs]),
            "branch_nothing_happened_rate": _rate(
                sum(bool(r.get("branch_nothing_happened")) for r in rs), n),
            "terminated": {t: sum(1 for r in rs if r.get("terminated") == t)
                           for t in sorted({r.get("terminated") for r in rs},
                                           key=str)},
        }

    out = {"by_rep_level_branch": {}, "paired_cost": {}}
    for (rep, lvl, branch), rs in sorted(by_group.items(), key=lambda kv: str(kv[0])):
        out["by_rep_level_branch"][f"{rep}|k{lvl}|{branch}"] = summarise(rs)

    # paired cost: same (state, rep, k), correct vs competitor branch
    pair_idx = defaultdict(dict)
    for r in rollouts:
        pair_idx[(r.get("state_id"), r.get("representation"), r.get("level"))][
            r.get("branch")] = r
    for (rep, lvl) in sorted({(r.get("representation"), r.get("level")) for r in rollouts},
                             key=str):
        extra_steps, both_succ, comp_fail, corr_fail, comp_recovers = [], 0, 0, 0, 0
        n_pairs = 0
        for key, d in pair_idx.items():
            if key[1] != rep or key[2] != lvl:
                continue
            c, m = d.get("correct"), d.get("competitor")
            if not c or not m:
                continue
            n_pairs += 1
            if c.get("success") and m.get("success"):
                both_succ += 1
                extra_steps.append(m["policy_steps_after_branch"]
                                   - c["policy_steps_after_branch"])
            elif c.get("success") and not m.get("success"):
                comp_fail += 1
            elif m.get("success") and not c.get("success"):
                corr_fail += 1
            if m.get("success"):
                comp_recovers += 1
        out["paired_cost"][f"{rep}|k{lvl}"] = {
            "n_pairs": n_pairs,
            "both_success": both_succ,
            "competitor_branch_failed": comp_fail,
            "correct_branch_failed": corr_fail,
            "competitor_success_rate": _rate(comp_recovers, n_pairs),
            "extra_steps_when_both_succeed": _mean_sem(extra_steps),
        }
    return out


# ══════════════════════════════════════════════════════════════════════
# QA
# ══════════════════════════════════════════════════════════════════════

def compute_qa(rows, docs, states, source_counts, rollouts):
    qa = {
        "n_cells": len(rows),
        "n_cells_not_fully_scored": sum(not r["fully_scored"] for r in rows),
        "n_cells_stem_prefix_mismatch": sum(not r["stem_prefix_ok"] for r in rows),
        "n_unparsable_generations": sum(r["unparsable"] for r in rows),
        "n_cells_competitor_not_admissible": sum(
            not r["competitor_admissible"] for r in rows
            if r["level"] is not None and r["level"] >= 0),
        "cells_by_rep": {rep: sum(r["representation"] == rep for r in rows)
                         for rep in sorted({r["representation"] for r in rows})},
        "cells_by_source_dir": source_counts,
        "levels_seen": sorted({r["level"] for r in rows}),
    }
    # The competitor MUST be in the state's admissible set — that is the whole
    # premise of this experiment.
    bad = []
    for r in rows:
        s = states.get(r["state_id"])
        if s and r["distractor_action"] not in [lex.normalize_action(a)
                                                for a in s.get("admissible_commands", [])]:
            bad.append(r["state_id"])
    qa["n_cells_competitor_not_in_admissible_set"] = len(set(bad))
    if docs:
        mismatched = []
        for (sid, lvl), text in docs.items():
            if lvl < 0:
                continue
            row = next((r for r in rows if r["state_id"] == sid
                        and r["level"] == lvl and r["representation"] == "text"), None)
            if row is None:
                continue
            kc = lex.count_word(text, row["distractor_keyword"])
            if kc != lvl:
                mismatched.append({"state_id": sid, "level": lvl,
                                   "keyword": row["distractor_keyword"],
                                   "measured": kc})
        qa["doc_keyword_count_mismatches"] = mismatched[:50]
        qa["n_doc_keyword_count_mismatches"] = len(mismatched)
    if rollouts:
        qa["n_rollouts"] = len(rollouts)
        qa["n_rollouts_failed_env"] = sum(not r.get("ok", True) for r in rollouts)
    return qa


# ══════════════════════════════════════════════════════════════════════
# Writers / summary
# ══════════════════════════════════════════════════════════════════════

PER_TRIAL_CSV_COLS = [
    "state_id", "level", "representation", "task_type", "step_idx",
    "correct_action", "distractor_action", "distractor_keyword",
    "competitor_admissible",
    "chosen_action", "chosen_verb", "chose_correct", "chose_distractor",
    "chosen_in_candidates", "unparsable", "is_error",
    "repro_keyword", "repro_exact", "repro_phrase", "repro_other_skillword",
    "argmax_sum", "argmax_is_correct", "argmax_is_distractor",
    "n_scored", "n_candidates",
    "p_correct_sum", "p_distractor_sum", "p_correct_mean", "p_distractor_mean",
    "entropy_sum", "entropy_norm_sum",
    "logp_correct_action", "logp_distractor_action", "logp_margin_action",
    "rank_correct_sum", "rank_distractor_sum",
]


def write_csv(rows, path, cols):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  wrote {path} ({len(rows)} rows)")


def write_by_level_rep_csv(by_level_rep, path):
    cols = (["level", "representation", "n"]
            + [f"mean_{m}" for m in DIST_METRICS]
            + ["error_rate", "chose_distractor_rate", "argmax_distractor_rate",
               "repro_keyword_rate", "fully_scored_rate"])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for lvl in sorted(by_level_rep, key=lambda x: int(x)):
            for rep in sorted(by_level_rep[lvl]):
                agg = by_level_rep[lvl][rep]
                w.writerow([lvl, rep, agg["n"]]
                           + [agg["means"][m]["mean"] for m in DIST_METRICS]
                           + [agg["rates"]["error_rate"],
                              agg["rates"]["chose_distractor_rate"],
                              agg["rates"]["argmax_distractor_rate"],
                              agg["rates"]["repro_keyword_rate"],
                              agg["rates"]["fully_scored_rate"]])
    print(f"  wrote {path}")


def print_summary(by_level_rep, levels, trends, bridge, paired, rollout):
    line = "=" * 118
    print("\n" + line)
    print("PER (k × representation) SUMMARY  —  mean ± s.e.m.   [competitor = admissible but wrong]")
    print(line)
    print(f"{'k':>3} {'rep':<11} {'n':>4} | {'P(comp)':>9} {'logP(comp)':>12} "
          f"{'margin':>9} {'rk(comp)':>8} | {'choseComp%':>10} {'argmaxComp%':>11} "
          f"{'err%':>6} {'reproKW%':>9}")
    for k in levels:
        for rep in MAIN_REPS:
            agg = by_level_rep.get(str(k), {}).get(rep)
            if not agg:
                continue
            m, r = agg["means"], agg["rates"]

            def ms(metric, nd=4):
                e = m.get(metric, {})
                return "n/a" if e.get("mean") is None else f"{e['mean']:.{nd}f}"
            rk = m["rank_distractor_sum"]["mean"]
            print(f"{k:>3} {rep:<11} {agg['n']:>4} | {ms('p_distractor_sum'):>9} "
                  f"{ms('logp_distractor_action'):>12} {ms('logp_margin_action', 3):>9} "
                  f"{(rk if rk is not None else float('nan')):8.2f} | "
                  f"{100 * (r['chose_distractor_rate'] or 0):10.1f} "
                  f"{100 * (r['argmax_distractor_rate'] or 0):11.1f} "
                  f"{100 * (r['error_rate'] or 0):6.1f} "
                  f"{100 * (r['repro_keyword_rate'] or 0) if r['repro_keyword_rate'] is not None else float('nan'):9.1f}")

    print("\n" + line)
    print("TRENDS vs k (aggregate OLS; the bridge claim: positive for BOTH "
          "P(competitor) and chose-competitor rate, steeper in Text)")
    print(line)
    for rep in MAIN_REPS:
        tr = trends.get(rep, {})
        if not tr:
            continue
        parts = []
        for metric in ("p_distractor_sum", "logp_distractor_action",
                       "chose_distractor_rate", "error_rate"):
            s = tr.get(metric, {}).get("slope")
            if s is not None:
                parts.append(f"{metric}={s:+.5f}/k")
        print(f"  {rep:<11} " + "  ".join(parts))
    dec = trends.get("slope_decomposition", {})
    for metric in ("p_distractor_sum", "chose_distractor_rate"):
        e = dec.get(metric, {})
        if e.get("total") is not None:
            print(f"  decomposition[{metric}]: total {e['total']:+.5f} = modality "
                  f"{e['modality']:+.5f} + structure {e['structure']:+.5f}"
                  + (f"  (modality share {e['modality_share']:.2f})"
                     if e.get("modality_share") is not None else ""))

    if bridge:
        print("\n" + line)
        print("PROBABILITY-TO-BEHAVIOR BRIDGE (cells where generation chose the "
              "competitor vs not)")
        print(line)
        for rep, e in bridge.items():
            pc, pn = e["p_distractor_when_chose"], e["p_distractor_when_not_chose"]
            pb = e["point_biserial_p_vs_choice"]
            print(f"  {rep:<11} P(comp | chose)={pc['mean'] if pc['mean'] is not None else float('nan'):.4f} "
                  f"(n={pc['n']})  vs  P(comp | not chose)="
                  f"{pn['mean'] if pn['mean'] is not None else float('nan'):.4f} (n={pn['n']})"
                  f"  | point-biserial r={pb.get('r')} (n={pb.get('n')})")
            if e.get("corr_delta_p_delta_chose", {}).get("r") is not None:
                print(f"              per-state Δp vs Δchose (k0→kmax): r="
                      f"{e['corr_delta_p_delta_chose']['r']:.3f} "
                      f"(n={e['n_states_both_ended']})")

    if paired:
        print("\n" + line)
        print("PAIRED k0 → kmax DELTAS (per state)")
        print(line)
        for metric, e in paired.get("metrics", {}).items():
            parts = []
            for rep in MAIN_REPS:
                d = e.get(rep, {})
                if d.get("mean") is not None:
                    parts.append(f"{rep} {d['mean']:+.4f}")
            t = e.get("total", {})
            tt = (f" | text−structural {t['mean']:+.4f} (p={t.get('sign_test_p')})"
                  if t.get("mean") is not None else "")
            print(f"  {metric:<24} " + "  ".join(parts) + tt)

    if rollout:
        print("\n" + line)
        print("ROLLOUT — end-to-end cost of executing the competitor")
        print(line)
        for key, e in rollout.get("paired_cost", {}).items():
            if not e["n_pairs"]:
                continue
            xs = e["extra_steps_when_both_succeed"]
            print(f"  [{key}] pairs={e['n_pairs']}: competitor-branch success "
                  f"{e['competitor_success_rate']:.1%}, competitor-branch failed "
                  f"{e['competitor_branch_failed']}, correct-branch failed "
                  f"{e['correct_branch_failed']}, extra steps when both succeed "
                  f"{xs['mean'] if xs['mean'] is not None else float('nan'):+.2f} "
                  f"± {xs['sem'] if xs['sem'] is not None else float('nan'):.2f} "
                  f"(n={xs['n']})")


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results_dirs = [args.results_dir.strip()]
    results_dirs += [d.strip() for d in (args.merge_results_dirs or "").split(",")
                     if d.strip()]
    raw, bad_files, source_counts = load_trials(results_dirs)
    if bad_files:
        print(f"[WARN] {len(bad_files)} unreadable result files, e.g. {bad_files[:3]}")
    if not raw:
        raise SystemExit(f"[ERROR] no probe results found in {results_dirs} — "
                         f"run run_probe.py first")
    states = load_states(args.states_dir) if args.states_dir else {}
    docs = load_documents(args.variants_dir) if args.variants_dir else {}
    rollouts = load_rollouts(args.rollout_dir)
    print(f"[INFO] loaded {len(raw)} cells | {len(states)} states | "
          f"{len(docs)} documents | {len(rollouts)} rollouts")

    rows = []
    for rec in raw:
        sid, lvl = rec.get("state_id"), rec.get("level")
        rows.append(flatten_trial(rec, states.get(sid), docs.get((sid, lvl))))

    levels = sorted({r["level"] for r in rows if r["level"] is not None and r["level"] >= 0})

    groups = defaultdict(list)
    for r in rows:
        groups[(r["level"], r["representation"])].append(r)
    by_level_rep = defaultdict(dict)
    for (lvl, rep), grp in groups.items():
        by_level_rep[str(lvl)][rep] = aggregate(grp)

    rep_groups = defaultdict(list)
    for r in rows:
        rep_groups[r["representation"]].append(r)
    pooled_by_rep = {rep: aggregate(grp) for rep, grp in rep_groups.items()}

    trends = compute_trends(by_level_rep, levels)
    per_state_trends = compute_per_state_trends(rows, levels)
    paired = compute_paired_deltas(rows, levels)
    bridge = compute_bridge(rows, levels)
    rollout_summary = compute_rollout_summary(rollouts, levels)
    qa = compute_qa(rows, docs, states, source_counts, rollouts)

    print_summary(by_level_rep, levels, trends, bridge, paired, rollout_summary)
    print("\n" + "=" * 118)
    print("QA")
    print("=" * 118)
    for k, v in qa.items():
        if k != "doc_keyword_count_mismatches":
            print(f"  {k}: {v}")
    if qa.get("n_cells_competitor_not_in_admissible_set"):
        print("  [FAIL] some competitors are NOT in their state's admissible set — "
              "wrong states dir? The bridge premise is broken for those cells.")

    analysis = {
        "meta": {
            "experiment": "probability-to-behavior",
            "n_cells": len(rows),
            "n_states": len({r['state_id'] for r in rows}),
            "levels": levels,
            "representations": sorted(rep_groups),
            "results_dirs": results_dirs,
            "rollout_dir": str(args.rollout_dir) if args.rollout_dir else None,
            "metric_definitions": {
                "competitor": "the distractor_action of this experiment: an "
                              "ADMISSIBLE (currently executable) but wrong/"
                              "premature action; chose_distractor is therefore "
                              "real wrong behaviour, not a hypothetical",
                "chose_distractor_rate": "free generation (temperature 0) "
                                         "picked the executable competitor",
                "argmax_distractor_rate": "forced-decoding argmax over the "
                                          "candidate set is the competitor",
                "bridge": "P(competitor) among cells that chose it vs cells "
                          "that did not + point-biserial correlation + "
                          "per-state Δp vs Δchoice correlation (k0→kmax)",
                "slope_decomposition": "total = slope(text) − slope(structural); "
                                       "modality = text − rasterized; structure = "
                                       "rasterized − structural",
                "logp_*_action": "forced-decoding log-probability, stem excluded",
                "rollout": "env rollout after the branch action with the model's "
                           "own greedy policy under the same skill conditioning; "
                           "extra_steps = competitor-branch steps − correct-branch "
                           "steps on pairs where both succeed",
            },
        },
        "qa": qa,
        "by_level_rep": dict(by_level_rep),
        "pooled_by_rep": pooled_by_rep,
        "trends": trends,
        "per_state_trends": per_state_trends,
        "paired_k0_kmax": paired,
        "bridge": bridge,
        "rollout": rollout_summary,
    }
    (out_dir / "probability_to_behavior_analysis.json").write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    print(f"\nSaved {out_dir / 'probability_to_behavior_analysis.json'}")

    write_csv(rows, out_dir / "per_trial.csv", PER_TRIAL_CSV_COLS)
    write_by_level_rep_csv(by_level_rep, out_dir / "by_level_rep.csv")

    # bridge_states.csv: per (state, rep) the k0→kmax probability and behaviour
    bridge_rows = []
    idx = {(r["state_id"], r["level"], r["representation"]): r for r in rows}
    for rep in MAIN_REPS:
        for sid in sorted({r["state_id"] for r in rows if r["representation"] == rep}):
            r0 = idx.get((sid, 0, rep))
            rk = idx.get((sid, max(levels), rep)) if levels else None
            if not r0 or not rk:
                continue
            bridge_rows.append({
                "state_id": sid, "representation": rep,
                "k_max": max(levels),
                "p_comp_k0": r0.get("p_distractor_sum"),
                "p_comp_kmax": rk.get("p_distractor_sum"),
                "delta_p": (rk["p_distractor_sum"] - r0["p_distractor_sum"])
                if (rk.get("p_distractor_sum") is not None
                    and r0.get("p_distractor_sum") is not None) else None,
                "chose_k0": r0.get("chose_distractor"),
                "chose_kmax": rk.get("chose_distractor"),
                "logp_comp_k0": r0.get("logp_distractor_action"),
                "logp_comp_kmax": rk.get("logp_distractor_action"),
            })
    write_csv(bridge_rows, out_dir / "bridge_states.csv",
              ["state_id", "representation", "k_max", "p_comp_k0", "p_comp_kmax",
               "delta_p", "chose_k0", "chose_kmax", "logp_comp_k0", "logp_comp_kmax"])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results")
    p.add_argument("--merge_results_dirs", default="",
                   help="optional comma-separated extra probe-result dirs")
    p.add_argument("--states_dir", default="states")
    p.add_argument("--variants_dir", default="variants")
    p.add_argument("--rollout_dir", default="rollout_results",
                   help="optional run_rollout.py output (skipped if absent)")
    p.add_argument("--output_dir", default="results/analysis")
    main(p.parse_args())
