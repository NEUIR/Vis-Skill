"""
Step 4 — analyse the lexical-interference probe results.

Implements the full chain the experiment is about:

    keyword-frequency interference (k)  →  probability change (P/logP/entropy)  →  final wrong behaviour (reproduction)

Inputs: the per-cell JSON files written by run_probe.py
(`{state_id}__k{level}__{rep}.json`), the extracted states (for queries and
history context), and — optionally — the constructed text variants (so the
"other skill word" reproduction check can read the actual document).

Analyses produced
=================

1. **Action distribution vs k** — for every (level k × representation):
     P(correct), P(distractor)          normalised over the candidate set
     logP(correct), logP(distractor)    forced-decoding log-probabilities,
                                        stem-excluded (`*_action`) and raw
     H = -Σ p_i log p_i                 action-distribution entropy (nats and
                                        normalised by log |candidates|)
   each as mean ± s.e.m., plus ranks and the forced-decoding argmax accuracy.

2. **Trends** — OLS slope of every metric against k, per representation, and
   the text-minus-visual slope difference. The claim under test is
   "the Text slope is clearly larger, Visual is flatter".

3. **Skill-word Reproduction Rate** — among the FREE-GENERATION errors
   (chosen action ≠ correct action):
     repro_keyword_rate  = #(error ∧ distractor keyword in chosen action) / #errors
     repro_exact_rate    = #(error ∧ chosen action == distractor action)  / #errors
     repro_phrase_rate   = #(error ∧ 'open microwave'-style verb phrase in
                            chosen action) / #errors
     repro_other_rate    = #(error ∧ chosen action contains a skill-document
                            word that occurs NOWHERE in the state's context
                            — query, observation history, correct action) / #errors
   The headline quantity of the report is `repro_keyword_rate`:
       Skill-word Reproduction Rate = #wrong outputs containing the Skill distractor word / #wrong outputs

4. **Bad-case quadrants** — pair the text and the visual cell of the same
   (state, k) and classify:
     both_correct / text_wrong_visual_right / text_right_visual_wrong /
     both_wrong
   For each discordant quadrant: how often the wrong side's action reproduces
   the skill distractor keyword. McNemar-style sign test (exact binomial, no
   scipy) on the discordant counts.

5. **Paired k=0 → k=8 deltas** — per state, ΔlogP(distractor), ΔlogP(correct),
   Δentropy, ΔP(distractor) for each representation, and the paired
   text-minus-visual difference with a sign test. This is the per-state version
   of the slope comparison and does not assume linearity in k.

6. **QA** — cells whose forced decoding was incomplete (n_scored <
   n_candidates), cells with a stem-tokenisation mismatch, unparsable
   generations, and the document-level keyword counts re-measured from the
   constructed variants (must equal k).

Outputs (under --output_dir):
  lexical_interference_analysis.json   everything, machine readable
  per_trial.csv                        one row per probed cell
  by_level_rep.csv                     the (k × representation) summary table
  bad_case_pairs.csv                   one row per paired (state, k)
  bad_cases_text_wrong_visual_right.json   qualitative dump for inspection
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

# Metrics averaged per (level, representation) cell. All are read from the
# probe result's `distribution` block; names match probe_lib.distribution_stats.
DIST_METRICS = [
    "p_correct_sum", "p_distractor_sum",
    "p_correct_mean", "p_distractor_mean",
    # `*_sum` softmaxes whole-sequence log-probs and therefore favours short
    # action strings; `*_mean` is the length-normalised counterpart. Both
    # entropies are kept so the entropy panel has a length-bias robustness check
    # rather than resting on the biased quantity alone.
    "entropy_sum", "entropy_norm_sum",
    "entropy_mean", "entropy_norm_mean",
    "logp_correct_action", "logp_distractor_action", "logp_margin_action",
    "logp_correct", "logp_distractor", "logp_margin",
    "rank_correct_sum", "rank_distractor_sum",
]

# Per-cell numeric metrics whose per-STATE slope against k is tested (see
# compute_per_state_trends). Rates are group-level, so they stay out of here and
# are covered by the aggregate trend in compute_trends.
PER_STATE_TREND_METRICS = [
    "logp_distractor_action", "logp_correct_action", "logp_margin_action",
    "entropy_sum", "p_distractor_sum", "p_correct_sum",
]

# Metrics whose trend against k is reported per representation.
TREND_METRICS = [
    "logp_distractor_action", "logp_correct_action", "logp_margin_action",
    "entropy_sum", "p_distractor_sum", "p_correct_sum",
    "error_rate", "chose_distractor_rate", "repro_keyword_rate",
]

# Metrics used for the paired per-state k0 -> kmax delta comparison.
DELTA_METRICS = ["logp_distractor_action", "logp_correct_action",
                 "entropy_sum", "p_distractor_sum"]

MAX_BAD_CASE_EXAMPLES = 300


# ══════════════════════════════════════════════════════════════════════
# Loading
# ══════════════════════════════════════════════════════════════════════

def load_trials(results_dir):
    """Load every probe cell written by run_probe.py."""
    trials, bad_files = [], []
    for f in sorted(Path(results_dir).glob(RESULT_GLOB)):
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            bad_files.append((f.name, str(e)))
            continue
        rec["_file"] = f.name
        trials.append(rec)
    return trials, bad_files


def load_states(states_dir):
    """-> {state_id: state dict} for context words and queries."""
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
    """-> {(state_id, level): document text} from the constructed text variants.

    The visual condition shows the SAME document as pixels, so the text
    variant is the right reference for skill-word attribution in both
    conditions.
    """
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


def context_words(state):
    """Words available to the model WITHOUT the skill document: the query, the
    whole observation history and the correct action. A wrong action built from
    a word outside this set but inside the document is skill-attributable."""
    if not state:
        return set()
    text = [state.get("query", ""), state.get("correct_action", "")]
    for msg in state.get("history_messages", []):
        c = msg.get("content", "")
        if isinstance(c, str):
            text.append(c)
    return set(lex.content_words(" ".join(text)))


# ══════════════════════════════════════════════════════════════════════
# Per-trial flattening
# ══════════════════════════════════════════════════════════════════════

def flatten_trial(rec, state, doc_text):
    """One probe cell -> one flat analysis record."""
    dist = rec.get("distribution") or {}
    gen = rec.get("generation") or {}

    chosen = gen.get("chosen_action", "") or ""
    keyword = rec.get("distractor_keyword", "")
    phrase = rec.get("distractor_phrase", "")

    chose_correct = bool(gen.get("chose_correct"))
    unparsable = not chosen
    is_error = (not chose_correct) and not unparsable   # parsable wrong action

    # ── reproduction flags (meaningful only on parsable errors) ──
    verb_surface = lex.VERB_SURFACE.get(rec.get("distractor_action_type", ""),
                                        rec.get("distractor_action_type", "") or "")
    repro_keyword = is_error and lex.count_word(chosen, keyword) > 0
    repro_exact = is_error and bool(gen.get("chose_distractor"))
    # Both grammars: contiguous "open microwave" AND interleaved
    # "clean candle 1 with sinkbasin 2". A contiguous-only test scored 0 for
    # every clean/heat/cool distractor even on a verbatim reproduction.
    repro_phrase = is_error and (
        lex.count_phrase(chosen, phrase) > 0
        or lex.count_verb_noun(chosen, verb_surface, keyword) > 0)
    repro_other = False
    if is_error and doc_text:
        doc_only = set(lex.content_words(doc_text)) - context_words(state)
        # This metric is about document words penetrating the action BEYOND the
        # injected keyword — that one is already measured by repro_keyword, so
        # leaving it in here would just double-count it.
        doc_only -= {keyword.lower()} | set(lex.content_words(keyword))
        doc_only -= set(lex.content_words(phrase))
        repro_other = any(w in doc_only for w in lex.content_words(chosen))

    row = {
        "state_id": rec.get("state_id"),
        "level": rec.get("level"),
        "representation": rec.get("representation"),
        "task_type": rec.get("task_type"),
        "step_idx": rec.get("step_idx"),
        "correct_action": rec.get("correct_action"),
        "distractor_action": rec.get("distractor_action"),
        "distractor_keyword": keyword,
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
# Aggregation helpers
# ══════════════════════════════════════════════════════════════════════

def _mean_sem(vals):
    vals = [v for v in vals if v is not None]
    n = len(vals)
    if n == 0:
        return {"mean": None, "sem": None, "std": None, "n": 0}
    mean = sum(vals) / n
    if n > 1:
        var = sum((v - mean) ** 2 for v in vals) / (n - 1)
        std = math.sqrt(var)
        sem = std / math.sqrt(n)
    else:
        std, sem = 0.0, 0.0
    return {"mean": mean, "sem": sem, "std": std, "n": n}


def _rate(num, den):
    return (num / den) if den else None


def aggregate(rows):
    """Rows of one (level, representation) group -> summary dict."""
    n = len(rows)
    counts = {
        "n_unparsable": sum(r["unparsable"] for r in rows),
        "n_error": sum(not r["chose_correct"] for r in rows),          # incl. unparsable
        "n_parsable_error": sum(r["is_error"] for r in rows),
        "n_chose_distractor": sum(r["chose_distractor"] for r in rows),
        "n_repro_keyword": sum(r["repro_keyword"] for r in rows),
        "n_repro_exact": sum(r["repro_exact"] for r in rows),
        "n_repro_phrase": sum(r["repro_phrase"] for r in rows),
        "n_repro_other_skillword": sum(r["repro_other_skillword"] for r in rows),
        "n_argmax_correct": sum(r["argmax_is_correct"] for r in rows),
        "n_fully_scored": sum(r["fully_scored"] for r in rows),
        "n_stem_prefix_ok": sum(r["stem_prefix_ok"] for r in rows),
    }
    pe = counts["n_parsable_error"]
    rates = {
        "correct_rate": _rate(n - counts["n_error"], n),
        "error_rate": _rate(counts["n_error"], n),
        "unparsable_rate": _rate(counts["n_unparsable"], n),
        "chose_distractor_rate": _rate(counts["n_chose_distractor"], n),
        "argmax_correct_rate": _rate(counts["n_argmax_correct"], n),
        "fully_scored_rate": _rate(counts["n_fully_scored"], n),
        # Skill-word Reproduction Rate and friends — denominator: parsable errors
        "repro_keyword_rate": _rate(counts["n_repro_keyword"], pe),
        "repro_exact_rate": _rate(counts["n_repro_exact"], pe),
        "repro_phrase_rate": _rate(counts["n_repro_phrase"], pe),
        "repro_other_skillword_rate": _rate(counts["n_repro_other_skillword"], pe),
    }
    means = {m: _mean_sem([r[m] for r in rows]) for m in DIST_METRICS}
    return {"n": n, "counts": counts, "rates": rates, "means": means}


def metric_value(agg, metric):
    """Unified accessor: `metric` may be a distribution mean or a rate."""
    if metric in agg["means"]:
        return agg["means"][metric]["mean"]
    if metric in agg["rates"]:
        return agg["rates"][metric]
    return None


def linear_trend(xs, ys):
    """OLS slope / intercept / R² (no scipy). xs, ys already aligned & finite."""
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
    return {
        "slope": slope,
        "intercept": ym - slope * xm,
        "r_squared": (ssxy ** 2) / (ssxx * ssyy) if ssyy > 0 else 0.0,
        "n_points": n,
    }


def sign_test_p(diffs):
    """Two-sided exact sign test on paired differences (no scipy)."""
    nz = [d for d in diffs if d is not None and d != 0]
    n = len(nz)
    if n == 0:
        return None
    k = sum(1 for d in nz if d > 0)
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) * (0.5 ** n)
    return min(1.0, 2 * tail)


# ══════════════════════════════════════════════════════════════════════
# Analyses
# ══════════════════════════════════════════════════════════════════════

def compute_trends(by_level_rep, levels):
    """OLS trend of each metric against k, per representation + slope diff."""
    trends = {}
    for rep in ("text", "visual"):
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
        trends[rep] = per_metric
    slope_diff = {}
    for metric in TREND_METRICS:
        ts = trends.get("text", {}).get(metric, {}).get("slope")
        vs = trends.get("visual", {}).get(metric, {}).get("slope")
        slope_diff[metric] = (ts - vs) if (ts is not None and vs is not None) else None
    trends["slope_difference_text_minus_visual"] = slope_diff
    return trends


def compute_per_state_trends(rows, levels):
    """Per-STATE OLS slope against k, per representation, plus the paired
    text-minus-visual slope difference with an exact sign test.

    `compute_trends` fits one line through the k aggregated cell means, so it
    yields a single slope with no uncertainty and supports no test. The claim
    under scrutiny — "the Text slope is clearly larger, Visual is flatter" — needs a distribution of
    slopes: regress each state's own k points, then compare the paired
    (text slope − visual slope) against 0. Unlike the k0→kmax delta this uses
    all levels, and unlike the aggregate slope it is inferential.
    """
    if len(levels) < 2:
        return {}
    idx = {}
    for r in rows:
        if r["representation"] in ("text", "visual"):
            idx[(r["state_id"], r["level"], r["representation"])] = r
    state_ids = sorted({sid for sid, _, _ in idx})

    out = {"levels": levels, "n_states": len(state_ids), "metrics": {}}
    for metric in PER_STATE_TREND_METRICS:
        per_rep = {}
        for rep in ("text", "visual"):
            slopes = {}
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
                        slopes[sid] = s
            per_rep[rep] = slopes
        diffs = [per_rep["text"][sid] - per_rep["visual"][sid]
                 for sid in sorted(per_rep["text"]) if sid in per_rep["visual"]]
        entry = {rep: _mean_sem(list(per_rep[rep].values())) for rep in ("text", "visual")}
        entry["diff_text_minus_visual"] = {
            **_mean_sem(diffs),
            "sign_test_p": sign_test_p(diffs),
            "n_positive": sum(1 for d in diffs if d > 0),
            "n_negative": sum(1 for d in diffs if d < 0),
        }
        out["metrics"][metric] = entry
    return out


def compute_bad_cases(rows, levels):
    """Pair text and visual cells of the same (state_id, level)."""
    by_pair = defaultdict(dict)
    for r in rows:
        if r["representation"] in ("text", "visual"):
            by_pair[(r["state_id"], r["level"])][r["representation"]] = r

    def classify(pair):
        t, v = pair.get("text"), pair.get("visual")
        if t is None or v is None:
            return None
        tc, vc = t["chose_correct"], v["chose_correct"]
        if tc and vc:
            return "both_correct"
        if not tc and vc:
            return "text_wrong_visual_right"
        if tc and not vc:
            return "text_right_visual_wrong"
        return "both_wrong"

    pair_rows = []
    for (sid, lvl), pair in sorted(by_pair.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        cat = classify(pair)
        if cat is None:
            continue
        t, v = pair["text"], pair["visual"]
        pair_rows.append({
            "state_id": sid,
            "level": lvl,
            "task_type": t["task_type"],
            "category": cat,
            "correct_action": t["correct_action"],
            "distractor_action": t["distractor_action"],
            "distractor_keyword": t["distractor_keyword"],
            "text_chosen": t["chosen_action"],
            "visual_chosen": v["chosen_action"],
            "text_repro_keyword": t["repro_keyword"],
            "visual_repro_keyword": v["repro_keyword"],
            "text_repro_exact": t["repro_exact"],
            "visual_repro_exact": v["repro_exact"],
            "text_repro_other_skillword": t["repro_other_skillword"],
            "visual_repro_other_skillword": v["repro_other_skillword"],
            "text_response_excerpt": t["response_excerpt"],
            "visual_response_excerpt": v["response_excerpt"],
        })

    def summarise(subset):
        n = len(subset)
        cats = defaultdict(int)
        for r in subset:
            cats[r["category"]] += 1
        tw = [r for r in subset if r["category"] == "text_wrong_visual_right"]
        vw = [r for r in subset if r["category"] == "text_right_visual_wrong"]
        bw = [r for r in subset if r["category"] == "both_wrong"]
        # discordant-pair sign test (McNemar-style, exact binomial)
        mcnemar_p = sign_test_p([1] * len(tw) + [-1] * len(vw))
        return {
            "n_pairs": n,
            "counts": dict(cats),
            "rates": {c: _rate(cats.get(c, 0), n) for c in
                      ("both_correct", "text_wrong_visual_right",
                       "text_right_visual_wrong", "both_wrong")},
            "text_wrong_visual_right": {
                "n": len(tw),
                "n_text_repro_keyword": sum(r["text_repro_keyword"] for r in tw),
                "text_repro_keyword_rate": _rate(
                    sum(r["text_repro_keyword"] for r in tw),
                    sum(1 for r in tw if r["text_chosen"])),
                "n_text_repro_exact": sum(r["text_repro_exact"] for r in tw),
                "text_repro_exact_rate": _rate(
                    sum(r["text_repro_exact"] for r in tw),
                    sum(1 for r in tw if r["text_chosen"])),
                "n_text_repro_other_skillword":
                    sum(r["text_repro_other_skillword"] for r in tw),
            },
            "text_right_visual_wrong": {
                "n": len(vw),
                "n_visual_repro_keyword": sum(r["visual_repro_keyword"] for r in vw),
                "visual_repro_keyword_rate": _rate(
                    sum(r["visual_repro_keyword"] for r in vw),
                    sum(1 for r in vw if r["visual_chosen"])),
                "n_visual_repro_exact": sum(r["visual_repro_exact"] for r in vw),
                "visual_repro_exact_rate": _rate(
                    sum(r["visual_repro_exact"] for r in vw),
                    sum(1 for r in vw if r["visual_chosen"])),
                "n_visual_repro_other_skillword":
                    sum(r["visual_repro_other_skillword"] for r in vw),
            },
            "both_wrong": {
                "n": len(bw),
                "text_repro_keyword_rate": _rate(
                    sum(r["text_repro_keyword"] for r in bw),
                    sum(1 for r in bw if r["text_chosen"])),
                "visual_repro_keyword_rate": _rate(
                    sum(r["visual_repro_keyword"] for r in bw),
                    sum(1 for r in bw if r["visual_chosen"])),
            },
            "mcnemar_sign_test_p": mcnemar_p,
        }

    out = {"pooled_all_levels": summarise(pair_rows)}
    if levels:
        kmax = max(levels)
        out["pooled_k_gt_0"] = summarise([r for r in pair_rows if r["level"] > 0])
        out[f"pooled_k_max_{kmax}"] = summarise(
            [r for r in pair_rows if r["level"] == kmax])
    out["by_level"] = {str(l): summarise([r for r in pair_rows if r["level"] == l])
                       for l in sorted({r["level"] for r in pair_rows})}
    return out, pair_rows


def compute_paired_deltas(rows, levels):
    """Per-state k=0 -> k=kmax delta of each metric, per representation, and
    the paired text-minus-visual difference with a sign test."""
    if not levels or max(levels) == 0:
        return {}
    kmax = max(levels)
    idx = {(r["state_id"], r["level"], r["representation"]): r for r in rows}
    state_ids = sorted({r["state_id"] for r in rows
                        if r["representation"] in ("text", "visual")})

    out = {"k_max": kmax, "n_states": len(state_ids), "metrics": {}}
    for metric in DELTA_METRICS:
        per_rep = {}
        diffs = []
        for rep in ("text", "visual"):
            vals = []
            for sid in state_ids:
                r0 = idx.get((sid, 0, rep))
                r8 = idx.get((sid, kmax, rep))
                if r0 and r8 and r0.get(metric) is not None and r8.get(metric) is not None:
                    vals.append((sid, r8[metric] - r0[metric]))
            per_rep[rep] = vals
        for sid, dt in per_rep["text"]:
            dv = dict(per_rep["visual"]).get(sid)
            if dv is not None:
                diffs.append(dt - dv)
        entry = {}
        for rep in ("text", "visual"):
            ds = [d for _, d in per_rep[rep]]
            entry[rep] = _mean_sem(ds)
        entry["diff_text_minus_visual"] = {
            **_mean_sem(diffs),
            "sign_test_p": sign_test_p(diffs),
            "n_positive": sum(1 for d in diffs if d > 0),
            "n_negative": sum(1 for d in diffs if d < 0),
        }
        out["metrics"][metric] = entry
    return out


def compute_qa(rows, docs):
    """Data-quality report."""
    qa = {
        "n_cells": len(rows),
        "n_cells_not_fully_scored": sum(not r["fully_scored"] for r in rows),
        "n_cells_stem_prefix_mismatch": sum(not r["stem_prefix_ok"] for r in rows),
        "n_unparsable_generations": sum(r["unparsable"] for r in rows),
        "cells_by_rep": {rep: sum(r["representation"] == rep for r in rows)
                         for rep in sorted({r["representation"] for r in rows})},
        "levels_seen": sorted({r["level"] for r in rows}),
    }
    # Re-measure keyword counts in the constructed documents (must equal k).
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
    return qa


def compute_by_task_type(rows):
    groups = defaultdict(list)
    for r in rows:
        if r["representation"] in ("text", "visual"):
            groups[(r["task_type"], r["representation"])].append(r)
    out = defaultdict(dict)
    for (tt, rep), grp in groups.items():
        agg = aggregate(grp)
        out[tt][rep] = {
            "n": agg["n"],
            "error_rate": agg["rates"]["error_rate"],
            "chose_distractor_rate": agg["rates"]["chose_distractor_rate"],
            "repro_keyword_rate": agg["rates"]["repro_keyword_rate"],
            "mean_logp_distractor_action": agg["means"]["logp_distractor_action"]["mean"],
            "mean_entropy_sum": agg["means"]["entropy_sum"]["mean"],
        }
    return dict(out)


# ══════════════════════════════════════════════════════════════════════
# Output writers
# ══════════════════════════════════════════════════════════════════════

PER_TRIAL_CSV_COLS = [
    "state_id", "level", "representation", "task_type", "step_idx",
    "correct_action", "distractor_action", "distractor_keyword",
    "chosen_action", "chosen_verb", "chose_correct", "chose_distractor",
    "chosen_in_candidates", "unparsable", "is_error",
    "repro_keyword", "repro_exact", "repro_phrase", "repro_other_skillword",
    "argmax_sum", "argmax_is_correct", "n_scored", "n_candidates",
    "p_correct_sum", "p_distractor_sum", "p_correct_mean", "p_distractor_mean",
    "entropy_sum", "entropy_norm_sum",
    "logp_correct_action", "logp_distractor_action", "logp_margin_action",
    "logp_correct", "logp_distractor",
    "rank_correct_sum", "rank_distractor_sum",
]

BAD_CASE_CSV_COLS = [
    "state_id", "level", "task_type", "category",
    "correct_action", "distractor_action", "distractor_keyword",
    "text_chosen", "visual_chosen",
    "text_repro_keyword", "visual_repro_keyword",
    "text_repro_exact", "visual_repro_exact",
    "text_repro_other_skillword", "visual_repro_other_skillword",
]


def write_per_trial_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PER_TRIAL_CSV_COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else
                            int(r[k]) if isinstance(r.get(k), bool) else r.get(k))
                        for k in PER_TRIAL_CSV_COLS})


def write_by_level_rep_csv(by_level_rep, path):
    cols = (["level", "representation", "n",
             "error_rate", "correct_rate", "chose_distractor_rate",
             "repro_keyword_rate", "repro_exact_rate", "repro_other_skillword_rate",
             "argmax_correct_rate"] +
            [f"{m}_mean" for m in DIST_METRICS] +
            [f"{m}_sem" for m in ("p_correct_sum", "p_distractor_sum",
                                  "entropy_sum", "logp_correct_action",
                                  "logp_distractor_action")])
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for lvl in sorted(by_level_rep, key=lambda s: int(s)):
            for rep in sorted(by_level_rep[lvl]):
                agg = by_level_rep[lvl][rep]
                row = [lvl, rep, agg["n"],
                       agg["rates"]["error_rate"], agg["rates"]["correct_rate"],
                       agg["rates"]["chose_distractor_rate"],
                       agg["rates"]["repro_keyword_rate"],
                       agg["rates"]["repro_exact_rate"],
                       agg["rates"]["repro_other_skillword_rate"],
                       agg["rates"]["argmax_correct_rate"]]
                row += [agg["means"][m]["mean"] for m in DIST_METRICS]
                row += [agg["means"][m]["sem"] for m in
                        ("p_correct_sum", "p_distractor_sum", "entropy_sum",
                         "logp_correct_action", "logp_distractor_action")]
                w.writerow(row)


def write_bad_case_csv(pair_rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=BAD_CASE_CSV_COLS, extrasaction="ignore")
        w.writeheader()
        for r in pair_rows:
            w.writerow({k: (int(r[k]) if isinstance(r.get(k), bool) else r.get(k))
                        for k in BAD_CASE_CSV_COLS})


# ══════════════════════════════════════════════════════════════════════
# Pretty printing
# ══════════════════════════════════════════════════════════════════════

def print_summary(by_level_rep, levels, trends, bad_cases, paired, per_state=None):
    line = "=" * 100
    print("\n" + line)
    print("ACTION DISTRIBUTION vs DISTRACTOR WORD FREQUENCY k  (mean over states)")
    print(line)
    hdr = (f"{'k':>3} {'rep':<7} {'N':>4} | {'P(corr)':>8} {'P(dist)':>8} "
           f"{'H':>6} | {'logP(corr)':>11} {'logP(dist)':>11} {'margin':>8} | "
           f"{'err%':>6} {'reproKW%':>9} {'=dist%':>7}")
    print(hdr)
    print("-" * 100)
    all_levels = sorted({int(l) for l in by_level_rep})
    for lvl in all_levels:
        for rep in sorted(by_level_rep[str(lvl)]):
            a = by_level_rep[str(lvl)][rep]
            m, rt = a["means"], a["rates"]

            def f(x, nd=4):
                return f"{x:.{nd}f}" if x is not None else "  n/a"

            def pct(x):
                return f"{100 * x:.1f}" if x is not None else "n/a"
            print(f"{lvl:>3} {rep:<7} {a['n']:>4} | "
                  f"{f(m['p_correct_sum']['mean']):>8} {f(m['p_distractor_sum']['mean']):>8} "
                  f"{f(m['entropy_sum']['mean'], 3):>6} | "
                  f"{f(m['logp_correct_action']['mean'], 3):>11} "
                  f"{f(m['logp_distractor_action']['mean'], 3):>11} "
                  f"{f(m['logp_margin_action']['mean'], 3):>8} | "
                  f"{pct(rt['error_rate']):>6} "
                  f"{pct(rt['repro_keyword_rate']):>9} "
                  f"{pct(rt['chose_distractor_rate']):>7}")

    print("\n" + line)
    print("TRENDS vs k  (OLS slope; the claim: |text slope| > |visual slope|)")
    print(line)
    print(f"{'metric':<26} {'text slope':>12} {'visual slope':>13} {'diff (T-V)':>12} "
          f"{'text R²':>8} {'vis R²':>8}")
    for metric in TREND_METRICS:
        t = trends.get("text", {}).get(metric, {})
        v = trends.get("visual", {}).get(metric, {})
        d = trends.get("slope_difference_text_minus_visual", {}).get(metric)

        def g(x):
            return f"{x:.5f}" if x is not None else "n/a"
        print(f"{metric:<26} {g(t.get('slope')):>12} {g(v.get('slope')):>13} "
              f"{g(d):>12} {g(t.get('r_squared')):>8} {g(v.get('r_squared')):>8}")

    if per_state and per_state.get("metrics"):
        print("\n" + line)
        print(f"PER-STATE SLOPES vs k  (n_states={per_state.get('n_states')}) — the "
              f"inferential version of the trend table above")
        print(line)
        print(f"{'metric':<26} {'text slope':>18} {'visual slope':>18} "
              f"{'T−V (paired)':>18} {'sign p':>9} {'+/−':>10}")
        for metric, e in per_state["metrics"].items():
            t, v, d = e["text"], e["visual"], e["diff_text_minus_visual"]

            def fm(s):
                return (f"{s['mean']:+.5f}±{s['sem']:.5f}" if s["mean"] is not None
                        else "n/a")
            p = d.get("sign_test_p")
            p_str = f"{p:.4g}" if p is not None else "n/a"
            pm_str = "+{}/-{}".format(d["n_positive"], d["n_negative"])
            print(f"{metric:<26} {fm(t):>18} {fm(v):>18} {fm(d):>18} "
                  f"{p_str:>9} {pm_str:>10}")

    pooled = bad_cases.get("pooled_all_levels", {})
    if pooled.get("n_pairs"):
        print("\n" + line)
        print("BAD-CASE QUADRANTS  (paired text vs visual on the same state × k)")
        print(line)
        for scope, bc in bad_cases.items():
            if scope == "by_level" or not bc.get("n_pairs"):
                continue
            c, rt = bc["counts"], bc["rates"]
            print(f"  [{scope}] n_pairs={bc['n_pairs']}")
            print(f"    both correct            : {c.get('both_correct', 0):>5}  "
                  f"({100 * (rt.get('both_correct') or 0):.1f}%)")
            print(f"    TEXT wrong / VIS right  : {c.get('text_wrong_visual_right', 0):>5}  "
                  f"({100 * (rt.get('text_wrong_visual_right') or 0):.1f}%)")
            print(f"    TEXT right / VIS wrong  : {c.get('text_right_visual_wrong', 0):>5}  "
                  f"({100 * (rt.get('text_right_visual_wrong') or 0):.1f}%)")
            print(f"    both wrong              : {c.get('both_wrong', 0):>5}  "
                  f"({100 * (rt.get('both_wrong') or 0):.1f}%)")
            tw = bc.get("text_wrong_visual_right", {})
            vw = bc.get("text_right_visual_wrong", {})
            if tw.get("n"):
                print(f"    → in TEXT-wrong cells: keyword reproduced "
                      f"{tw.get('n_text_repro_keyword', 0)}/{tw['n']} "
                      f"({100 * (tw.get('text_repro_keyword_rate') or 0):.1f}%), "
                      f"exact distractor "
                      f"({100 * (tw.get('text_repro_exact_rate') or 0):.1f}%)")
            if vw.get("n"):
                print(f"    → in VIS-wrong cells : keyword reproduced "
                      f"{vw.get('n_visual_repro_keyword', 0)}/{vw['n']} "
                      f"({100 * (vw.get('visual_repro_keyword_rate') or 0):.1f}%), "
                      f"exact distractor "
                      f"({100 * (vw.get('visual_repro_exact_rate') or 0):.1f}%)")
            p = bc.get("mcnemar_sign_test_p")
            print(f"    discordant-pair sign test p = {p:.4f}" if p is not None
                  else "    discordant-pair sign test p = n/a")

    if paired and paired.get("metrics"):
        print("\n" + line)
        print(f"PAIRED PER-STATE DELTAS  k=0 → k={paired.get('k_max')}  "
              f"(n_states={paired.get('n_states')})")
        print(line)
        for metric, e in paired["metrics"].items():
            t, v, d = e["text"], e["visual"], e["diff_text_minus_visual"]

            def fm(s):
                return (f"{s['mean']:+.4f} ± {s['sem']:.4f} (n={s['n']})"
                        if s["mean"] is not None else "n/a")
            print(f"  Δ {metric}:")
            print(f"      text   : {fm(t)}")
            print(f"      visual : {fm(v)}")
            print(f"      T−V    : {fm(d)}  sign-test p="
                  f"{d['sign_test_p']:.4g}  (+{d['n_positive']}/−{d['n_negative']})"
                  if d.get("sign_test_p") is not None else
                  f"      T−V    : {fm(d)}")


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw, bad_files = load_trials(args.results_dir)
    if bad_files:
        print(f"[WARN] {len(bad_files)} unreadable result files, e.g. {bad_files[:3]}")
    if not raw:
        raise SystemExit(f"[ERROR] no probe results found in {args.results_dir} "
                         f"(glob {RESULT_GLOB}) — run run_probe.py first")
    states = load_states(args.states_dir) if args.states_dir else {}
    docs = load_documents(args.variants_dir) if args.variants_dir else {}
    print(f"[INFO] loaded {len(raw)} cells | {len(states)} states | {len(docs)} documents")

    rows = []
    for rec in raw:
        sid = rec.get("state_id")
        lvl = rec.get("level")
        # text and visual cells of the same (state, level) show the SAME
        # document (pixels vs characters), so skill-word attribution reads the
        # text variant for both; control cells (level -1) have no document.
        doc_text = docs.get((sid, lvl))
        rows.append(flatten_trial(rec, states.get(sid), doc_text))

    levels = sorted({r["level"] for r in rows if r["level"] is not None and r["level"] >= 0})

    # ── 1. (level × representation) aggregation ──
    groups = defaultdict(list)
    for r in rows:
        groups[(r["level"], r["representation"])].append(r)
    by_level_rep = defaultdict(dict)
    for (lvl, rep), grp in groups.items():
        by_level_rep[str(lvl)][rep] = aggregate(grp)

    pooled_by_rep = {}
    rep_groups = defaultdict(list)
    for r in rows:
        rep_groups[r["representation"]].append(r)
    for rep, grp in rep_groups.items():
        pooled_by_rep[rep] = aggregate(grp)

    # ── 2. trends (aggregate slope + the inferential per-state version) ──
    trends = compute_trends(by_level_rep, levels)
    per_state_trends = compute_per_state_trends(rows, levels)

    # ── 3. bad cases ──
    bad_cases, pair_rows = compute_bad_cases(rows, levels)

    # ── 4. paired deltas ──
    paired = compute_paired_deltas(rows, levels)

    # ── 5. breakdowns & QA ──
    by_task_type = compute_by_task_type(rows)
    qa = compute_qa(rows, docs)

    # ── print ──
    print_summary(by_level_rep, levels, trends, bad_cases, paired, per_state_trends)
    print("\n" + "=" * 100)
    print("QA")
    print("=" * 100)
    for k, v in qa.items():
        if k != "doc_keyword_count_mismatches":
            print(f"  {k}: {v}")
    if qa.get("n_doc_keyword_count_mismatches"):
        print(f"  [WARN] keyword-count mismatches, e.g. "
              f"{qa['doc_keyword_count_mismatches'][:5]}")

    # ── save ──
    analysis = {
        "meta": {
            "n_cells": len(rows),
            "n_states": len({r['state_id'] for r in rows}),
            "levels": levels,
            "representations": sorted(rep_groups),
            "results_dir": str(args.results_dir),
            "states_dir": str(args.states_dir) if args.states_dir else None,
            "variants_dir": str(args.variants_dir) if args.variants_dir else None,
            "metric_definitions": {
                "P(correct)/P(distractor)":
                    "softmax over forced-decoded candidate log-probs "
                    "(*_sum: whole-sequence; *_mean: length-normalised)",
                "logp_*_action":
                    "forced-decoding log-probability of the action tokens only "
                    "(constant 'Thought: ...\\nAction:' stem excluded)",
                "entropy_sum":
                    "H = -Σ p_i log p_i over the candidate set (nats), from the "
                    "whole-sequence softmax (length-biased toward short actions)",
                "entropy_mean":
                    "same H but from the length-normalised (per-action-token) "
                    "softmax — the length-bias robustness check",
                "per_state_trends":
                    "per-state OLS slope vs k; diff_text_minus_visual + exact "
                    "sign test is the inferential form of the slope claim",
                "error_rate":
                    "free generation did not produce the trajectory's correct action "
                    "(unparsable included)",
                "repro_keyword_rate":
                    "Skill-word Reproduction Rate = parsable errors whose action "
                    "contains the distractor keyword / all parsable errors",
                "repro_exact_rate":
                    "parsable errors whose action == the distractor action / all "
                    "parsable errors",
                "repro_other_skillword_rate":
                    "parsable errors whose action contains a document word absent "
                    "from query+history+correct action / all parsable errors",
            },
        },
        "qa": qa,
        "by_level_rep": dict(by_level_rep),
        "pooled_by_rep": pooled_by_rep,
        "trends": trends,
        "per_state_trends": per_state_trends,
        "paired_k0_kmax": paired,
        "bad_cases": bad_cases,
        "by_task_type": by_task_type,
    }
    (out_dir / "lexical_interference_analysis.json").write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")

    write_per_trial_csv(rows, out_dir / "per_trial.csv")
    write_by_level_rep_csv(dict(by_level_rep), out_dir / "by_level_rep.csv")
    write_bad_case_csv(pair_rows, out_dir / "bad_case_pairs.csv")

    examples = [r for r in pair_rows
                if r["category"] == "text_wrong_visual_right"][:MAX_BAD_CASE_EXAMPLES]
    (out_dir / "bad_cases_text_wrong_visual_right.json").write_text(
        json.dumps(examples, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nSaved:")
    print(f"  {out_dir / 'lexical_interference_analysis.json'}")
    print(f"  {out_dir / 'per_trial.csv'}")
    print(f"  {out_dir / 'by_level_rep.csv'}")
    print(f"  {out_dir / 'bad_case_pairs.csv'}")
    print(f"  {out_dir / 'bad_cases_text_wrong_visual_right.json'} "
          f"({len(examples)} examples)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", required=True,
                   help="directory with run_probe.py cell outputs")
    p.add_argument("--states_dir", default=None,
                   help="extracted states (context words for skill attribution)")
    p.add_argument("--variants_dir", default=None,
                   help="constructed variants (document text for skill-word checks)")
    p.add_argument("--output_dir", required=True)
    main(p.parse_args())
