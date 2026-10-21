"""
Step 4 — analyse the causal keyword ablation.

The causal question: is it this specific word in the Skill that drives the
corresponding action prediction, or a generic change of context?

Evidence structure (all paired per state, so document/state difficulty cancels):

    Δcf(metric) = metric(original) − metric(counterfactual)
        removing the distractor keyword. If the keyword is causal, Δcf > 0 for
        logP(distractor) / P(distractor) and Δcf < 0 for rank(distractor).

    Δmc(metric) = metric(original) − metric(matched_control)
        removing a frequency-matched (and where possible position-matched)
        UNRELATED word. "Deleting any word changes the probability" predicts Δmc ≈ Δcf; the causal
        claim predicts Δmc ≈ 0.

    DiD = Δcf − Δmc
        the difference-in-differences — the headline causal statistic, with an
        exact sign test over states.

    Δcf(text) − Δcf(structural)
        the modality claim: after removing the keyword, the Text distractor probability drops more.

Free generation contributes the behavioural mirror: chose-distractor rates per
version and the paired discordant counts (original chose it, counterfactual
did not — and the matched-control analogue).

Outputs (under --output_dir):
  causal_ablation_analysis.json   everything, machine readable
  per_trial.csv                   one row per probed cell
  by_version_rep.csv              the (k × version × representation) table
  per_state_deltas.csv            one row per (state, k, rep, metric) triple
"""
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
REP_ALIAS = {"visual": "structural"}
MAIN_REPS = ("text", "structural", "rasterized")
VERSIONS = ("original", "counterfactual", "matched_control")

DIST_METRICS = [
    "p_correct_sum", "p_distractor_sum",
    "p_correct_mean", "p_distractor_mean",
    "entropy_sum", "entropy_norm_sum",
    "logp_correct_action", "logp_distractor_action", "logp_margin_action",
    "rank_correct_sum", "rank_distractor_sum",
]

# Metrics carried through the paired delta / DiD machinery.
DELTA_METRICS = [
    "logp_distractor_action", "p_distractor_sum", "rank_distractor_sum",
    "logp_margin_action", "logp_correct_action", "entropy_sum",
]
# For these, "the keyword pushed the distractor" means Δ(original − ablated) > 0.
HIGHER_IS_MORE_DISTRACTED = {
    "logp_distractor_action": True, "p_distractor_sum": True,
    "rank_distractor_sum": False, "logp_margin_action": False,
    "logp_correct_action": None, "entropy_sum": None,
}


# ══════════════════════════════════════════════════════════════════════
# Loading / flattening
# ══════════════════════════════════════════════════════════════════════

def load_trials(results_dir):
    trials, bad_files = [], []
    for f in sorted(Path(results_dir).glob(RESULT_GLOB)):
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            bad_files.append((f.name, str(e)))
            continue
        if "version" not in rec:
            # A lexical-interference style cell (no version factor) is not part
            # of this experiment; skip loudly rather than mis-group it.
            bad_files.append((f.name, "missing 'version' field"))
            continue
        rec["representation"] = REP_ALIAS.get(rec.get("representation"),
                                              rec.get("representation"))
        rec["_file"] = f.name
        trials.append(rec)
    return trials, bad_files


def flatten_trial(rec):
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
    row = {
        "state_id": rec.get("state_id"),
        "level": rec.get("level"),
        "version": rec.get("version"),
        "representation": rec.get("representation"),
        "task_type": rec.get("task_type"),
        "correct_action": rec.get("correct_action"),
        "distractor_action": rec.get("distractor_action"),
        "distractor_keyword": keyword,
        "control_word": rec.get("control_word"),
        "control_replacement": rec.get("control_replacement"),
        "keyword_replacement": rec.get("keyword_replacement"),
        "chosen_action": chosen,
        "chose_correct": chose_correct,
        "chose_distractor": bool(gen.get("chose_distractor")),
        "unparsable": unparsable,
        "is_error": is_error,
        "repro_keyword": is_error and lex.count_word(chosen, keyword) > 0,
        "repro_phrase": is_error and (
            lex.count_phrase(chosen, phrase) > 0
            or lex.count_verb_noun(chosen, verb_surface, keyword) > 0),
        "argmax_sum": dist.get("argmax_sum", ""),
        "argmax_is_distractor": dist.get("argmax_sum") == rec.get("distractor_action"),
        "n_scored": dist.get("n_scored", 0),
        "n_candidates": dist.get("n_candidates", 0),
        "fully_scored": bool(dist.get("n_scored")) and
                        dist.get("n_scored") == dist.get("n_candidates"),
        "stem_prefix_ok": bool(dist.get("stem_prefix_ok", True)),
        "doc_keyword_count": rec.get("doc_keyword_count"),
        "response_excerpt": (gen.get("response_text", "") or "")[:240],
    }
    for m in DIST_METRICS:
        row[m] = dist.get(m)
    return row


# ══════════════════════════════════════════════════════════════════════
# Stats helpers (no scipy)
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


def sign_test_p(diffs):
    nz = [d for d in diffs if d is not None and d != 0]
    n = len(nz)
    if n == 0:
        return None
    k = sum(1 for d in nz if d > 0)
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) * (0.5 ** n)
    return min(1.0, 2 * tail)


def _diff_summary(diffs, positive_means=None):
    """paired-difference summary; `positive_means` states the sign that supports
    'the keyword was pushing the distractor' (True: >0, False: <0, None: n/a)."""
    out = _mean_sem(diffs)
    out.update({
        "sign_test_p": sign_test_p(diffs),
        "n_positive": sum(1 for d in diffs if d is not None and d > 0),
        "n_negative": sum(1 for d in diffs if d is not None and d < 0),
    })
    if positive_means is True:
        out["n_consistent"] = out["n_positive"]
    elif positive_means is False:
        out["n_consistent"] = out["n_negative"]
    return out


def aggregate(rows):
    n = len(rows)
    counts = {
        "n_unparsable": sum(r["unparsable"] for r in rows),
        "n_error": sum(not r["chose_correct"] for r in rows),
        "n_chose_distractor": sum(r["chose_distractor"] for r in rows),
        "n_repro_keyword": sum(r["repro_keyword"] for r in rows),
        "n_argmax_distractor": sum(r["argmax_is_distractor"] for r in rows),
        "n_fully_scored": sum(r["fully_scored"] for r in rows),
        "n_stem_prefix_ok": sum(r["stem_prefix_ok"] for r in rows),
    }
    pe = sum(r["is_error"] for r in rows)
    rates = {
        "correct_rate": _rate(n - counts["n_error"], n),
        "error_rate": _rate(counts["n_error"], n),
        "chose_distractor_rate": _rate(counts["n_chose_distractor"], n),
        "argmax_distractor_rate": _rate(counts["n_argmax_distractor"], n),
        "repro_keyword_rate": _rate(counts["n_repro_keyword"], pe),
        "fully_scored_rate": _rate(counts["n_fully_scored"], n),
    }
    means = {m: _mean_sem([r[m] for r in rows]) for m in DIST_METRICS}
    return {"n": n, "counts": counts, "rates": rates, "means": means}


# ══════════════════════════════════════════════════════════════════════
# Core analyses
# ══════════════════════════════════════════════════════════════════════

def compute_paired_ablation(rows, levels, reps):
    """Per-state Δcf / Δmc / DiD for every (level, rep, metric)."""
    idx = {(r["state_id"], r["level"], r["version"], r["representation"]): r
           for r in rows}
    out = {"levels": levels, "reps": reps, "cells": {}}
    delta_rows = []
    for lvl in levels:
        for rep in reps:
            sids = sorted({r["state_id"] for r in rows
                           if r["level"] == lvl and r["representation"] == rep})
            for metric in DELTA_METRICS:
                pos = HIGHER_IS_MORE_DISTRACTED.get(metric)
                dcf, dmc = [], []
                per_state = {}
                for sid in sids:
                    o = idx.get((sid, lvl, "original", rep))
                    c = idx.get((sid, lvl, "counterfactual", rep))
                    m = idx.get((sid, lvl, "matched_control", rep))
                    if o and c and o.get(metric) is not None and c.get(metric) is not None:
                        v_cf = o[metric] - c[metric]
                        dcf.append(v_cf)
                    else:
                        v_cf = None
                    if o and m and o.get(metric) is not None and m.get(metric) is not None:
                        v_mc = o[metric] - m[metric]
                        dmc.append(v_mc)
                    else:
                        v_mc = None
                    did = (v_cf - v_mc) if (v_cf is not None and v_mc is not None) else None
                    per_state[sid] = {"delta_cf": v_cf, "delta_mc": v_mc, "did": did}
                    delta_rows.append({
                        "state_id": sid, "level": lvl, "representation": rep,
                        "metric": metric, "delta_cf": v_cf, "delta_mc": v_mc,
                        "did": did,
                    })
                key = f"k{lvl}__{rep}"
                out["cells"].setdefault(key, {})[metric] = {
                    "delta_cf": _diff_summary(dcf, pos),
                    "delta_mc": _diff_summary(dmc, pos),
                    "did": _diff_summary([s["did"] for s in per_state.values()], pos),
                    "per_state": per_state,
                }
    out["delta_rows"] = delta_rows
    return out


def compute_cross_rep(rows, levels, reps):
    """Δcf(text) − Δcf(structural): does the TEXT condition drop more when the
    keyword is removed? (paired per state)."""
    idx = {(r["state_id"], r["level"], r["version"], r["representation"]): r
           for r in rows}
    out = {}
    pairs = [(a, b) for i, a in enumerate(reps) for b in reps[i + 1:]]
    for lvl in levels:
        for metric in DELTA_METRICS:
            entry = {}
            for a, b in pairs:
                diffs = []
                sids = sorted({r["state_id"] for r in rows if r["level"] == lvl})
                for sid in sids:
                    vals = []
                    for rep in (a, b):
                        o = idx.get((sid, lvl, "original", rep))
                        c = idx.get((sid, lvl, "counterfactual", rep))
                        vals.append(o[metric] - c[metric]
                                    if (o and c and o.get(metric) is not None
                                        and c.get(metric) is not None) else None)
                    if vals[0] is not None and vals[1] is not None:
                        diffs.append(vals[0] - vals[1])
                entry[f"{a}_minus_{b}"] = _diff_summary(diffs)
            out[f"k{lvl}__{metric}"] = entry
    return out


def compute_behavior_pairs(rows, levels, reps):
    """Free-generation mirror: paired chose-distractor outcomes, original vs
    each ablated version (McNemar-style discordant counts + exact sign test)."""
    idx = {(r["state_id"], r["level"], r["version"], r["representation"]): r
           for r in rows}
    out = {}
    for lvl in levels:
        for rep in reps:
            for version in ("counterfactual", "matched_control"):
                both_chose = only_orig = only_ver = neither = 0
                sids = sorted({r["state_id"] for r in rows if r["level"] == lvl})
                for sid in sids:
                    o = idx.get((sid, lvl, "original", rep))
                    v = idx.get((sid, lvl, version, rep))
                    if not o or not v:
                        continue
                    oc, vc = o["chose_distractor"], v["chose_distractor"]
                    if oc and vc:
                        both_chose += 1
                    elif oc and not vc:
                        only_orig += 1
                    elif vc and not oc:
                        only_ver += 1
                    else:
                        neither += 1
                n = both_chose + only_orig + only_ver + neither
                out[f"k{lvl}__{rep}__{version}"] = {
                    "n_pairs": n,
                    "both_chose_distractor": both_chose,
                    "only_original_chose": only_orig,
                    f"only_{version}_chose": only_ver,
                    "neither_chose": neither,
                    "orig_rate": _rate(both_chose + only_orig, n),
                    "version_rate": _rate(both_chose + only_ver, n),
                    "mcnemar_sign_test_p": sign_test_p(
                        [1] * only_orig + [-1] * only_ver),
                }
    return out


def compute_qa(rows, construction_meta):
    qa = {
        "n_cells": len(rows),
        "cells_by_rep_version": defaultdict(int),
        "n_cells_not_fully_scored": sum(not r["fully_scored"] for r in rows),
        "n_cells_stem_prefix_mismatch": sum(not r["stem_prefix_ok"] for r in rows),
        "n_unparsable_generations": sum(r["unparsable"] for r in rows),
    }
    for r in rows:
        qa["cells_by_rep_version"][f"{r['representation']}|{r['version']}"] += 1
    qa["cells_by_rep_version"] = dict(qa["cells_by_rep_version"])
    # The documents must be what they claim: re-measure keyword counts per version.
    bad_kw = [r["state_id"] for r in rows
              if r["version"] == "original" and r["doc_keyword_count"] != r["level"]]
    bad_cf = [r["state_id"] for r in rows
              if r["version"] == "counterfactual" and (r["doc_keyword_count"] or 0) != 0]
    bad_mc = [r["state_id"] for r in rows
              if r["version"] == "matched_control"
              and r["doc_keyword_count"] != r["level"]]
    qa["n_original_cells_keyword_count_wrong"] = len(bad_kw)
    qa["n_counterfactual_cells_keyword_present"] = len(bad_cf)
    qa["n_matched_control_cells_keyword_count_wrong"] = len(bad_mc)
    if construction_meta:
        qa["construction_problems"] = construction_meta.get("problems", {})
        tiers = defaultdict(int)
        for m in construction_meta.get("states", []):
            for lvl, rec in m.get("levels", {}).items():
                mc = rec.get("versions", {}).get("matched_control", {})
                tiers[f"k{lvl}_tier{mc.get('tier', 'none')}"] += 1
        qa["control_word_tiers"] = dict(sorted(tiers.items()))
    return qa


# ══════════════════════════════════════════════════════════════════════
# Writers / summary
# ══════════════════════════════════════════════════════════════════════

PER_TRIAL_CSV_COLS = [
    "state_id", "level", "version", "representation", "task_type",
    "correct_action", "distractor_action", "distractor_keyword",
    "control_word", "control_replacement", "keyword_replacement",
    "chosen_action", "chose_correct", "chose_distractor", "unparsable",
    "is_error", "repro_keyword", "repro_phrase",
    "argmax_sum", "argmax_is_distractor", "n_scored", "n_candidates",
    "doc_keyword_count",
    "p_correct_sum", "p_distractor_sum",
    "p_correct_mean", "p_distractor_mean",
    "logp_correct_action", "logp_distractor_action", "logp_margin_action",
    "rank_correct_sum", "rank_distractor_sum", "entropy_sum",
]


def write_csv(rows, path, cols):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  wrote {path} ({len(rows)} rows)")


def write_by_version_rep_csv(by_lvr, path):
    cols = (["level", "version", "representation", "n"]
            + [f"mean_{m}" for m in DIST_METRICS]
            + ["error_rate", "chose_distractor_rate", "argmax_distractor_rate",
               "repro_keyword_rate", "fully_scored_rate"])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for lvl in sorted(by_lvr, key=int):
            for version in VERSIONS:
                for rep in sorted(by_lvr[lvl]):
                    agg = by_lvr[lvl].get((version, rep))
                    if not agg:
                        continue
                    w.writerow([lvl, version, rep, agg["n"]]
                               + [agg["means"][m]["mean"] for m in DIST_METRICS]
                               + [agg["rates"]["error_rate"],
                                  agg["rates"]["chose_distractor_rate"],
                                  agg["rates"]["argmax_distractor_rate"],
                                  agg["rates"]["repro_keyword_rate"],
                                  agg["rates"]["fully_scored_rate"]])
    print(f"  wrote {path}")


def print_summary(by_lvr, levels, reps, paired, cross, behavior):
    line = "=" * 116
    print("\n" + line)
    print("PER (k × version × representation) SUMMARY  —  mean ± s.e.m.")
    print(line)
    print(f"{'k':>2} {'version':<16} {'rep':<11} {'n':>4} | {'logP(dist)':>13} "
          f"{'P(dist)':>9} {'rank(dist)':>10} {'margin':>9} | {'choseD%':>8} {'err%':>6}")
    for lvl in levels:
        for version in VERSIONS:
            for rep in reps:
                agg = by_lvr.get(str(lvl), {}).get((version, rep))
                if not agg:
                    continue
                m, r = agg["means"], agg["rates"]

                def ms(metric, nd=4):
                    e = m.get(metric, {})
                    if e.get("mean") is None:
                        return "n/a"
                    return f"{e['mean']:.{nd}f}±{e['sem']:.4f}" if e.get("sem") else f"{e['mean']:.{nd}f}"
                print(f"{lvl:>2} {version:<16} {rep:<11} {agg['n']:>4} | "
                      f"{ms('logp_distractor_action'):>13} {ms('p_distractor_sum'):>9} "
                      f"{ms('rank_distractor_sum', 2):>10} {ms('logp_margin_action', 3):>9} | "
                      f"{100 * (r['chose_distractor_rate'] or 0):8.1f} "
                      f"{100 * (r['error_rate'] or 0):6.1f}")

    print("\n" + line)
    print("PAIRED ABLATION EFFECTS (original − ablated; per state, exact sign tests)")
    print(line)
    for key, metrics in paired.get("cells", {}).items():
        print(f"  [{key}]")
        for metric, e in metrics.items():
            def fmt(d):
                if d.get("mean") is None:
                    return "n/a"
                return (f"{d['mean']:+.4f}±{d['sem']:.4f} (n={d['n']}, "
                        f"p={d['sign_test_p']:.3g})" if d.get("sign_test_p") is not None
                        else f"{d['mean']:+.4f}±{d['sem']:.4f} (n={d['n']})")
            print(f"    {metric:<24} Δcf {fmt(e['delta_cf']):<38} "
                  f"Δmc {fmt(e['delta_mc']):<38} DiD {fmt(e['did'])}")

    print("\n" + line)
    print("CROSS-REPRESENTATION: Δcf(a) − Δcf(b)  (does Text drop more?)")
    print(line)
    for key, pairs in cross.items():
        print(f"  [{key}]")
        for name, d in pairs.items():
            if d.get("mean") is None:
                continue
            print(f"    {name:<28} {d['mean']:+.4f} ± {d['sem']:.4f} "
                  f"(n={d['n']}, +{d['n_positive']}/−{d['n_negative']}, "
                  f"sign-test p={d.get('sign_test_p')})")

    print("\n" + line)
    print("FREE-GENERATION BEHAVIOUR (chose-distractor, paired per state)")
    print(line)
    for key, e in behavior.items():
        if not e["n_pairs"]:
            continue
        print(f"  [{key}] n={e['n_pairs']}: orig {e['orig_rate']:.1%} vs "
              f"ablated {e['version_rate']:.1%} | only-orig {e['only_original_chose']}, "
              f"only-ablated {e.get('only_counterfactual_chose', e.get('only_matched_control_chose'))}, "
              f"both {e['both_chose_distractor']} | McNemar p={e['mcnemar_sign_test_p']}")


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw, bad_files = load_trials(args.results_dir)
    if bad_files:
        print(f"[WARN] {len(bad_files)} unusable result files, e.g. {bad_files[:3]}")
    if not raw:
        raise SystemExit(f"[ERROR] no probe results in {args.results_dir} "
                         f"(glob {RESULT_GLOB}) — run run_probe.py first")
    rows = [flatten_trial(rec) for rec in raw]
    print(f"[INFO] loaded {len(rows)} cells")

    levels = sorted({r["level"] for r in rows})
    reps = [rep for rep in MAIN_REPS
            if any(r["representation"] == rep for r in rows)]

    groups = defaultdict(list)
    for r in rows:
        groups[(r["level"], r["version"], r["representation"])].append(r)
    by_lvr = defaultdict(dict)
    for (lvl, version, rep), grp in groups.items():
        by_lvr[str(lvl)][(version, rep)] = aggregate(grp)

    construction_meta = None
    cmeta_file = Path(args.variants_dir) / "construction_meta.json" if args.variants_dir else None
    if cmeta_file and cmeta_file.exists():
        construction_meta = json.loads(cmeta_file.read_text(encoding="utf-8"))

    paired = compute_paired_ablation(rows, levels, reps)
    cross = compute_cross_rep(rows, levels, reps)
    behavior = compute_behavior_pairs(rows, levels, reps)
    qa = compute_qa(rows, construction_meta)

    print_summary(by_lvr, levels, reps, paired, cross, behavior)
    print("\n" + "=" * 116)
    print("QA")
    print("=" * 116)
    for k, v in qa.items():
        print(f"  {k}: {v}")

    delta_rows = paired.pop("delta_rows")
    analysis = {
        "meta": {
            "experiment": "causal-keyword-ablation",
            "n_cells": len(rows),
            "n_states": len({r['state_id'] for r in rows}),
            "levels": levels,
            "versions": list(VERSIONS),
            "representations": reps,
            "results_dir": str(args.results_dir),
            "metric_definitions": {
                "delta_cf": "metric(original) − metric(counterfactual): the "
                            "keyword-removal effect, paired per state",
                "delta_mc": "metric(original) − metric(matched_control): the "
                            "frequency-matched unrelated-word removal",
                "did": "Δcf − Δmc — the causal statistic; 'delete any word' predicts ≈0, "
                       "the keyword-causal claim predicts >0 (for "
                       "logp_distractor_action / p_distractor_sum)",
                "cross_rep": "Δcf(a) − Δcf(b): whether TEXT loses more distractor "
                             "probability than the image conditions when the "
                             "keyword is removed",
                "logp_*_action": "forced-decoding log-probability, stem excluded",
            },
        },
        "qa": qa,
        "by_level_version_rep": {lvl: {f"{v}|{r}": agg for (v, r), agg in d.items()}
                                 for lvl, d in by_lvr.items()},
        "paired_ablation": paired,
        "cross_rep_deltas": cross,
        "behavior_pairs": behavior,
    }
    (out_dir / "causal_ablation_analysis.json").write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    print(f"\nSaved {out_dir / 'causal_ablation_analysis.json'}")

    write_csv(rows, out_dir / "per_trial.csv", PER_TRIAL_CSV_COLS)
    write_by_version_rep_csv(by_lvr, out_dir / "by_version_rep.csv")
    write_csv(delta_rows, out_dir / "per_state_deltas.csv",
              ["state_id", "level", "representation", "metric",
               "delta_cf", "delta_mc", "did"])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results")
    p.add_argument("--variants_dir", default="variants")
    p.add_argument("--output_dir", default="results/analysis")
    main(p.parse_args())
