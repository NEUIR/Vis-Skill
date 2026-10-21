"""
Step 4 — analyse the rasterized-interference probe results.

Question: where does Visual's robustness to interference come from?

    MODALITY  — "the words no longer enter as language tokens" (text -> pixels)
    STRUCTURE — "structured layout: titles, sections, hierarchy, indentation" (plain pixels -> skill cards)

Design: the lexical-interference measurement with a THIRD representation arm.
Same skill documents, same k = 0/1/2/4/8, same decision states, same candidate
actions, same forced-decoding path:

    text        document as characters
    rasterized  same document as plain-text-screenshot cards (modality only)
    structural  same document as styled skill cards          (modality + structure)

Inputs: probe cells from one or more result directories, in precedence order
(`--results_dirs`, comma separated). By default the NEW rasterized cells come
from ./results and the already-measured text/visual cells are merged from
../../lexical-interference/results (representation "visual" is normalised to
"structural"; on duplicate (state, level, rep) keys the EARLIER directory wins,
so local re-measurements override merged ones). Merging is sound only because
run_probe.py/probe_lib.py here are the lexical-interference measurement path
unchanged — the QA block re-checks that merged cells carry the same distractor
fields and candidate sets as the states on disk.

Analyses produced
=================

1. **(level × representation) aggregation** for text / rasterized / structural
   (+ the no-skill control): P(correct), P(distractor), logP(correct),
   logP(distractor), margin, entropy, ranks, argmax accuracy, free-generation
   error / chose-distractor / reproduction rates — mean ± s.e.m.

2. **Slope decomposition** (the headline). For every trend metric, per-state
   OLS slopes against k are computed for each representation and decomposed:

       total     = slope(text)       − slope(structural)   (the LI contrast)
       modality  = slope(text)       − slope(rasterized)   (chars -> plain pixels)
       structure = slope(rasterized) − slope(structural)   (plain -> styled pixels)
       total = modality + structure  (exactly, per state and in aggregate)

   with exact sign tests on each component and share statistics
   (modality_share = modality/total). Verdicts:
       both arms flat           -> no interference effect to decompose
       modality_share  >= 0.7   -> mainly the visual modality itself weakens the direct lexical pull
       structure_share >= 0.7   -> what really matters is the structured organisation
       otherwise                -> both modality and structure contribute (rasterized lies in between)

3. **Bad-case quadrants** for each representation pair (text-vs-rasterized,
   structural-vs-rasterized, text-vs-structural): both_correct / a_wrong_b_right
   / a_right_b_wrong / both_wrong, reproduction rates in the discordant
   quadrants, McNemar-style exact sign test.

4. **Paired k=0 → k=kmax deltas** per representation and the paired pairwise
   differences with sign tests (the distribution-free mirror of 2).

5. **QA** — incomplete forced decodings, stem mismatches, unparsable
   generations, document keyword-count re-measurement (must equal k), merged
   cell counts per source directory, and a check that merged cells agree with
   the states on distractor action / candidate set.

Outputs (under --output_dir):
  rasterized_interference_analysis.json   everything, machine readable
  per_trial.csv                           one row per probed cell
  by_level_rep.csv                        the (k × representation) summary table
  bad_case_pairs.csv                      one row per paired (state, k, pair)
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

# "visual" is the lexical-interference name of the structural cards; normalise
# on load so every downstream group key is one of MAIN_REPS (+ "none").
REP_ALIAS = {"visual": "structural"}
MAIN_REPS = ("text", "rasterized", "structural")

# Decomposition pairs: (a, b, component name). slope(a) − slope(b).
DECOMPOSITION = (
    ("text", "structural", "total"),
    ("text", "rasterized", "modality"),
    ("rasterized", "structural", "structure"),
)

BAD_CASE_PAIRS = (
    ("text", "rasterized"),
    ("structural", "rasterized"),
    ("text", "structural"),
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
]

DELTA_METRICS = ["logp_distractor_action", "logp_correct_action",
                 "entropy_sum", "p_distractor_sum"]

MAX_BAD_CASE_EXAMPLES = 300


# ══════════════════════════════════════════════════════════════════════
# Loading
# ══════════════════════════════════════════════════════════════════════

def load_trials(results_dirs):
    """Load probe cells from every directory, in precedence order.

    -> (trials, bad_files, source_counts). A cell key is
    (state_id, level, normalised representation); the FIRST directory holding a
    key wins, so a local re-measurement overrides a merged historical cell.
    """
    trials, bad_files = [], []
    source_counts = defaultdict(int)
    seen = set()
    for d in results_dirs:
        d = Path(d)
        if not d.exists():
            print(f"[WARN] results dir not found, skipping: {d}")
            continue
        for f in sorted(d.glob(RESULT_GLOB)):
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
    """-> {(state_id, level): document text} from the constructed text variants.

    All three representations show the SAME document (characters / plain pixels
    / styled pixels), so the text variant is the reference for skill-word
    attribution in every condition.
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
    """One probe cell -> one flat analysis record (identical to the
    lexical-interference flattening, plus the source-directory field)."""
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
    n = len(rows)
    counts = {
        "n_unparsable": sum(r["unparsable"] for r in rows),
        "n_error": sum(not r["chose_correct"] for r in rows),
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
        "repro_keyword_rate": _rate(counts["n_repro_keyword"], pe),
        "repro_exact_rate": _rate(counts["n_repro_exact"], pe),
        "repro_phrase_rate": _rate(counts["n_repro_phrase"], pe),
        "repro_other_skillword_rate": _rate(counts["n_repro_other_skillword"], pe),
    }
    means = {m: _mean_sem([r[m] for r in rows]) for m in DIST_METRICS}
    return {"n": n, "counts": counts, "rates": rates, "means": means}


def metric_value(agg, metric):
    if metric in agg["means"]:
        return agg["means"][metric]["mean"]
    if metric in agg["rates"]:
        return agg["rates"][metric]
    return None


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
    return {
        "slope": slope,
        "intercept": ym - slope * xm,
        "r_squared": (ssxy ** 2) / (ssxx * ssyy) if ssyy > 0 else 0.0,
        "n_points": n,
    }


def sign_test_p(diffs):
    nz = [d for d in diffs if d is not None and d != 0]
    n = len(nz)
    if n == 0:
        return None
    k = sum(1 for d in nz if d > 0)
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) * (0.5 ** n)
    return min(1.0, 2 * tail)


def _diff_summary(diffs):
    return {
        **_mean_sem(diffs),
        "sign_test_p": sign_test_p(diffs),
        "n_positive": sum(1 for d in diffs if d is not None and d > 0),
        "n_negative": sum(1 for d in diffs if d is not None and d < 0),
    }


# ══════════════════════════════════════════════════════════════════════
# Analyses
# ══════════════════════════════════════════════════════════════════════

def compute_trends(by_level_rep, levels):
    """Aggregate OLS trend of each metric against k, per representation, plus
    the slope decomposition total = modality + structure."""
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


def decomposition_verdict(entry, eps=1e-4):
    """Automatic reading of one metric's slope decomposition."""
    tot, mod, stru = entry.get("total"), entry.get("modality"), entry.get("structure")
    if tot is None or mod is None or stru is None:
        return "insufficient data"
    if abs(tot) < eps:
        return "no text-vs-image slope difference to decompose"
    sm, ss = mod / tot, stru / tot
    if sm >= 0.7:
        return "modality-dominant (the visual modality itself weakens the direct lexical pull)"
    if ss >= 0.7:
        return "structure-dominant (the structured organisation is what matters)"
    if sm > 0.3 and ss > 0.3:
        return "mixed (both modality and structure contribute, rasterized lies in between)"
    if sm > ss:
        return "modality-leaning mixed"
    return "structure-leaning mixed"


def compute_per_state_trends(rows, levels):
    """Per-STATE OLS slope against k for each representation, and the paired
    decomposition with exact sign tests:

        modality  diff = slope(text)       − slope(rasterized)
        structure diff = slope(rasterized) − slope(structural)
        total     diff = slope(text)       − slope(structural)

    This is the inferential form of the aggregate decomposition: every state
    contributes one paired triple, so the sign tests are exact and the shares
    are computed per state as well as on the means.
    """
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
        # Per-state shares, defined where the state's total slope is non-trivial.
        shares_m, shares_s = [], []
        for sid in sorted(slopes["text"]):
            if sid in slopes["rasterized"] and sid in slopes["structural"]:
                tot = slopes["text"][sid] - slopes["structural"][sid]
                if abs(tot) > 1e-6:
                    shares_m.append((slopes["text"][sid] - slopes["rasterized"][sid]) / tot)
                    shares_s.append((slopes["rasterized"][sid] - slopes["structural"][sid]) / tot)
        entry["modality_share"] = _mean_sem(shares_m)
        entry["structure_share"] = _mean_sem(shares_s)
        out["metrics"][metric] = entry
    return out


def compute_bad_cases(rows, levels):
    """Quadrant analysis for every representation pair of interest."""
    by_key = defaultdict(dict)
    for r in rows:
        if r["representation"] in MAIN_REPS:
            by_key[(r["state_id"], r["level"])][r["representation"]] = r

    out = {}
    pair_rows = []
    for a, b in BAD_CASE_PAIRS:
        pair_name = f"{a}_vs_{b}"

        def classify(pair):
            ra, rb = pair.get(a), pair.get(b)
            if ra is None or rb is None:
                return None
            ac, bc = ra["chose_correct"], rb["chose_correct"]
            if ac and bc:
                return "both_correct"
            if not ac and bc:
                return "a_wrong_b_right"
            if ac and not bc:
                return "a_right_b_wrong"
            return "both_wrong"

        rows_pair = []
        for (sid, lvl), pair in sorted(by_key.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            cat = classify(pair)
            if cat is None:
                continue
            ra, rb = pair[a], pair[b]
            rows_pair.append({
                "pair": pair_name,
                "state_id": sid,
                "level": lvl,
                "task_type": ra["task_type"],
                "category": cat,
                "correct_action": ra["correct_action"],
                "distractor_action": ra["distractor_action"],
                "distractor_keyword": ra["distractor_keyword"],
                "a_chosen": ra["chosen_action"],
                "b_chosen": rb["chosen_action"],
                "a_repro_keyword": ra["repro_keyword"],
                "b_repro_keyword": rb["repro_keyword"],
                "a_repro_exact": ra["repro_exact"],
                "b_repro_exact": rb["repro_exact"],
                "a_repro_other_skillword": ra["repro_other_skillword"],
                "b_repro_other_skillword": rb["repro_other_skillword"],
                "a_response_excerpt": ra["response_excerpt"],
                "b_response_excerpt": rb["response_excerpt"],
            })
        pair_rows.extend(rows_pair)

        def summarise(subset):
            n = len(subset)
            cats = defaultdict(int)
            for r in subset:
                cats[r["category"]] += 1
            aw = [r for r in subset if r["category"] == "a_wrong_b_right"]
            bw = [r for r in subset if r["category"] == "a_right_b_wrong"]
            bb = [r for r in subset if r["category"] == "both_wrong"]
            mcnemar_p = sign_test_p([1] * len(aw) + [-1] * len(bw))
            return {
                "a": a, "b": b,
                "n_pairs": n,
                "counts": dict(cats),
                "rates": {c: _rate(cats.get(c, 0), n) for c in
                          ("both_correct", "a_wrong_b_right",
                           "a_right_b_wrong", "both_wrong")},
                "a_wrong_b_right": {
                    "n": len(aw),
                    "n_a_repro_keyword": sum(r["a_repro_keyword"] for r in aw),
                    "a_repro_keyword_rate": _rate(
                        sum(r["a_repro_keyword"] for r in aw),
                        sum(1 for r in aw if r["a_chosen"])),
                    "n_a_repro_exact": sum(r["a_repro_exact"] for r in aw),
                    "a_repro_exact_rate": _rate(
                        sum(r["a_repro_exact"] for r in aw),
                        sum(1 for r in aw if r["a_chosen"])),
                },
                "a_right_b_wrong": {
                    "n": len(bw),
                    "n_b_repro_keyword": sum(r["b_repro_keyword"] for r in bw),
                    "b_repro_keyword_rate": _rate(
                        sum(r["b_repro_keyword"] for r in bw),
                        sum(1 for r in bw if r["b_chosen"])),
                    "n_b_repro_exact": sum(r["b_repro_exact"] for r in bw),
                    "b_repro_exact_rate": _rate(
                        sum(r["b_repro_exact"] for r in bw),
                        sum(1 for r in bw if r["b_chosen"])),
                },
                "both_wrong": {
                    "n": len(bb),
                    "a_repro_keyword_rate": _rate(
                        sum(r["a_repro_keyword"] for r in bb),
                        sum(1 for r in bb if r["a_chosen"])),
                    "b_repro_keyword_rate": _rate(
                        sum(r["b_repro_keyword"] for r in bb),
                        sum(1 for r in bb if r["b_chosen"])),
                },
                "mcnemar_sign_test_p": mcnemar_p,
            }

        pair_out = {"pooled_all_levels": summarise(rows_pair)}
        if levels:
            kmax = max(levels)
            pair_out[f"pooled_k_max_{kmax}"] = summarise(
                [r for r in rows_pair if r["level"] == kmax])
        pair_out["by_level"] = {str(l): summarise([r for r in rows_pair if r["level"] == l])
                                for l in sorted({r["level"] for r in rows_pair})}
        out[pair_name] = pair_out
    return out, pair_rows


def compute_paired_deltas(rows, levels):
    """Per-state k=0 -> k=kmax delta per representation + the paired pairwise
    differences (the distribution-free mirror of the slope decomposition)."""
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


def compute_qa(rows, docs, states, source_counts):
    qa = {
        "n_cells": len(rows),
        "n_cells_not_fully_scored": sum(not r["fully_scored"] for r in rows),
        "n_cells_stem_prefix_mismatch": sum(not r["stem_prefix_ok"] for r in rows),
        "n_unparsable_generations": sum(r["unparsable"] for r in rows),
        "cells_by_rep": {rep: sum(r["representation"] == rep for r in rows)
                         for rep in sorted({r["representation"] for r in rows})},
        "cells_by_source_dir": source_counts,
        "levels_seen": sorted({r["level"] for r in rows}),
    }
    # Merged cells must describe the SAME measurement as the states on disk:
    # same distractor action and same candidate set. A mismatch means the
    # merged directory belongs to a different extraction and must not be used.
    inconsistent = []
    for r in rows:
        s = states.get(r["state_id"])
        if not s:
            continue
        if (r["distractor_action"] != lex.normalize_action(s["distractor_action"])
                or r["n_candidates"] != len(s["candidate_actions"])):
            inconsistent.append(r["state_id"])
    qa["n_merged_state_inconsistencies"] = len(set(inconsistent))
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
        if r["representation"] in MAIN_REPS:
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
    "state_id", "level", "representation", "source_dir", "task_type", "step_idx",
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
    "pair", "state_id", "level", "task_type", "category",
    "correct_action", "distractor_action", "distractor_keyword",
    "a_chosen", "b_chosen", "a_repro_keyword", "b_repro_keyword",
    "a_repro_exact", "b_repro_exact",
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
            + ["error_rate", "chose_distractor_rate", "repro_keyword_rate",
               "repro_exact_rate", "argmax_correct_rate", "fully_scored_rate"])
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
                              agg["rates"]["repro_keyword_rate"],
                              agg["rates"]["repro_exact_rate"],
                              agg["rates"]["argmax_correct_rate"],
                              agg["rates"]["fully_scored_rate"]])
    print(f"  wrote {path}")


# ══════════════════════════════════════════════════════════════════════
# Console summary
# ══════════════════════════════════════════════════════════════════════

def _fmt(v, nd=4):
    return "  n/a " if v is None else f"{v:+.{nd}f}" if nd else f"{v:.{nd}f}"


def print_summary(by_level_rep, levels, trends, per_state, bad_cases, paired):
    line = "=" * 112
    print("\n" + line)
    print("PER (k × representation) SUMMARY  —  mean ± s.e.m.")
    print(line)
    header = (f"{'k':>3} {'rep':<11} {'n':>4} | {'logP(dist)':>12} {'logP(corr)':>12} "
              f"{'margin':>9} | {'P(dist)':>8} {'P(corr)':>8} {'H':>7} | "
              f"{'rk(dist)':>8} | {'err%':>6} {'choseD%':>8} {'reproKW%':>9}")
    print(header)
    for k in levels:
        for rep in MAIN_REPS:
            agg = by_level_rep.get(str(k), {}).get(rep)
            if not agg:
                continue
            m = agg["means"]
            r = agg["rates"]

            def ms(metric, nd=4):
                e = m.get(metric, {})
                if e.get("mean") is None:
                    return " " * nd
                return f"{e['mean']:.{nd}f}±{e['sem']:.4f}" if e.get("sem") else f"{e['mean']:.{nd}f}"
            print(f"{k:>3} {rep:<11} {agg['n']:>4} | {ms('logp_distractor_action'):>12} "
                  f"{ms('logp_correct_action', 3):>12} {ms('logp_margin_action', 3):>9} | "
                  f"{ms('p_distractor_sum'):>8} {ms('p_correct_sum', 3):>8} "
                  f"{ms('entropy_sum', 3):>7} | "
                  f"{(m['rank_distractor_sum']['mean'] or 0):8.2f} | "
                  f"{100 * (r['error_rate'] or 0):6.1f} "
                  f"{100 * (r['chose_distractor_rate'] or 0):8.1f} "
                  f"{100 * (r['repro_keyword_rate'] or 0) if r['repro_keyword_rate'] is not None else float('nan'):9.1f}")

    print("\n" + line)
    print("SLOPE DECOMPOSITION (aggregate OLS slopes vs k)")
    print(line)
    dec = trends.get("slope_decomposition", {})
    print(f"{'metric':<26} {'s(text)':>10} {'s(raster)':>10} {'s(struct)':>10} | "
          f"{'total':>10} {'modality':>10} {'structure':>10} | {'mod.shr':>8} verdict")
    for metric in TREND_METRICS:
        e = dec.get(metric)
        if not e:
            continue
        s = e["slopes"]
        print(f"{metric:<26} {_fmt(s.get('text'), 5):>10} {_fmt(s.get('rasterized'), 5):>10} "
              f"{_fmt(s.get('structural'), 5):>10} | {_fmt(e.get('total'), 5):>10} "
              f"{_fmt(e.get('modality'), 5):>10} {_fmt(e.get('structure'), 5):>10} | "
              f"{_fmt(e.get('modality_share'), 3):>8} {decomposition_verdict(e)}")

    if per_state:
        print("\n" + line)
        print("PER-STATE SLOPE DECOMPOSITION (paired, exact sign tests)")
        print(line)
        for metric, e in per_state.get("metrics", {}).items():
            print(f"  {metric}:")
            for name in ("total", "modality", "structure"):
                d = e.get(name, {})
                if d.get("mean") is None:
                    continue
                print(f"    {name:<10} mean {d['mean']:+.5f} ± {d['sem']:.5f}  "
                      f"(n={d['n']}, +/{d['n_positive']}/−{d['n_negative']}, "
                      f"sign-test p={d.get('sign_test_p')})")
            sm = e.get("modality_share", {})
            if sm.get("mean") is not None:
                print(f"    per-state modality share {sm['mean']:.3f} ± {sm['sem']:.3f} "
                      f"(n={sm['n']})")

    print("\n" + line)
    print("PAIRED k0 → kmax DELTAS")
    print(line)
    for metric, e in paired.get("metrics", {}).items():
        print(f"  {metric}:")
        for rep in MAIN_REPS:
            d = e.get(rep, {})
            if d.get("mean") is None:
                continue
            print(f"    {rep:<11} Δ(kmax−k0) {d['mean']:+.4f} ± {d['sem']:.4f} (n={d['n']})")
        for name in ("total", "modality", "structure"):
            d = e.get(name, {})
            if d.get("mean") is None:
                continue
            print(f"    {name:<11} paired diff {d['mean']:+.4f} ± {d['sem']:.4f} "
                  f"(n={d['n']}, sign-test p={d.get('sign_test_p')})")

    print("\n" + line)
    print("BAD-CASE QUADRANTS (pooled over all k)")
    print(line)
    for pair_name, po in bad_cases.items():
        p = po["pooled_all_levels"]
        if not p["n_pairs"]:
            continue
        a, b = p["a"], p["b"]
        print(f"  {a} vs {b} (n={p['n_pairs']}): "
              f"both_correct {p['rates']['both_correct']:.1%} | "
              f"{a}_wrong_{b}_right {p['rates']['a_wrong_b_right']:.1%} "
              f"(n={p['a_wrong_b_right']['n']}, KW repro "
              f"{p['a_wrong_b_right']['a_repro_keyword_rate'] or 0:.1%}) | "
              f"{a}_right_{b}_wrong {p['rates']['a_right_b_wrong']:.1%} "
              f"(n={p['a_right_b_wrong']['n']}, KW repro "
              f"{p['a_right_b_wrong']['b_repro_keyword_rate'] or 0:.1%}) | "
              f"both_wrong {p['rates']['both_wrong']:.1%} | "
              f"McNemar p={p['mcnemar_sign_test_p']}")


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
        raise SystemExit(f"[ERROR] no probe results found in {results_dirs} "
                         f"(glob {RESULT_GLOB}) — run run_probe.py first")
    states = load_states(args.states_dir) if args.states_dir else {}
    docs = load_documents(args.variants_dir) if args.variants_dir else {}
    print(f"[INFO] loaded {len(raw)} cells | {len(states)} states | {len(docs)} documents")
    for d, n in source_counts.items():
        print(f"[INFO]   {n:5d} cells from {d}")
    reps_present = {r["representation"] for r in raw}
    for rep in ("text", "rasterized", "structural"):
        if rep not in reps_present:
            print(f"[WARN] representation '{rep}' has NO cells — the decomposition "
                  f"needs all three (text/structural merge from the "
                  f"lexical-interference results dir; rasterized from run_probe.py)")

    rows = []
    for rec in raw:
        sid = rec.get("state_id")
        lvl = rec.get("level")
        doc_text = docs.get((sid, lvl))
        rows.append(flatten_trial(rec, states.get(sid), doc_text))

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
    bad_cases, pair_rows = compute_bad_cases(rows, levels)
    paired = compute_paired_deltas(rows, levels)
    by_task_type = compute_by_task_type(rows)
    qa = compute_qa(rows, docs, states, source_counts)

    print_summary(by_level_rep, levels, trends, per_state_trends, bad_cases, paired)
    print("\n" + "=" * 112)
    print("QA")
    print("=" * 112)
    for k, v in qa.items():
        if k != "doc_keyword_count_mismatches":
            print(f"  {k}: {v}")
    if qa.get("n_doc_keyword_count_mismatches"):
        print(f"  [WARN] keyword-count mismatches, e.g. "
              f"{qa['doc_keyword_count_mismatches'][:5]}")
    if qa.get("n_merged_state_inconsistencies"):
        print(f"  [FAIL] merged cells disagree with the states on disk — the merged "
              f"results dir belongs to a DIFFERENT extraction; do not interpret.")

    analysis = {
        "meta": {
            "experiment": "rasterized-interference",
            "n_cells": len(rows),
            "n_states": len({r['state_id'] for r in rows}),
            "levels": levels,
            "representations": sorted(rep_groups),
            "results_dirs": results_dirs,
            "states_dir": str(args.states_dir) if args.states_dir else None,
            "variants_dir": str(args.variants_dir) if args.variants_dir else None,
            "metric_definitions": {
                "representations":
                    "text = document as characters; rasterized = same document as "
                    "plain-text-screenshot cards (modality change only); "
                    "structural = same document as styled skill cards (modality + "
                    "structure); 'visual' in lexical-interference files is the "
                    "same condition as 'structural'",
                "slope_decomposition":
                    "total = slope(text) − slope(structural); modality = "
                    "slope(text) − slope(rasterized); structure = "
                    "slope(rasterized) − slope(structural); total = modality + "
                    "structure exactly",
                "P(correct)/P(distractor)":
                    "softmax over forced-decoded candidate log-probs "
                    "(*_sum: whole-sequence; *_mean: length-normalised)",
                "logp_*_action":
                    "forced-decoding log-probability of the action tokens only "
                    "(constant 'Thought: ...\\nAction:' stem excluded)",
                "entropy_sum":
                    "H = -Σ p_i log p_i over the candidate set (nats)",
                "per_state_trends":
                    "per-state OLS slope vs k; paired component differences with "
                    "exact sign tests are the inferential form of the "
                    "decomposition",
                "error_rate":
                    "free generation did not produce the trajectory's correct action "
                    "(unparsable included)",
                "repro_keyword_rate":
                    "Skill-word Reproduction Rate = parsable errors whose action "
                    "contains the distractor keyword / all parsable errors",
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
    (out_dir / "rasterized_interference_analysis.json").write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    print(f"\nSaved {out_dir / 'rasterized_interference_analysis.json'}")

    write_csv(rows, out_dir / "per_trial.csv", PER_TRIAL_CSV_COLS)
    write_by_level_rep_csv(by_level_rep, out_dir / "by_level_rep.csv")
    write_csv(pair_rows, out_dir / "bad_case_pairs.csv", BAD_CASE_CSV_COLS)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results",
                   help="primary (local) results dir — highest merge precedence")
    p.add_argument("--merge_results_dirs", default="../lexical-interference/results",
                   help="comma-separated additional dirs to merge text/visual "
                        "cells from (empty string to disable)")
    p.add_argument("--states_dir", default="states")
    p.add_argument("--variants_dir", default="variants",
                   help="dir whose text/ subfolder holds the constructed documents")
    p.add_argument("--output_dir", default="results/analysis")
    main(p.parse_args())
