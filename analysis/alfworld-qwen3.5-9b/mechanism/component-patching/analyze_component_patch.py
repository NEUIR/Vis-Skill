#!/usr/bin/env python3
"""
component-patching analyzer — shared by both models (the Gemma tree's copy is a symlink).

Turns per-cell JSONs into the Recovered exposure effect of README §6:

    G = S_{k=8}(a_D) - S_{k=0}(a_D)          exposure gap
    R = S_patch(a_D) - S_{k=0}(a_D)          what the patch recovered
    recovered % = 100 * mean(R) / mean(G)    RATIO OF MEANS, game-clustered CI

Never a mean of per-state ratios: states with a near-zero G make that diverge,
and 165 states come from only 64 games, so a state-level bootstrap would
understate the CI. The ratio is recomputed inside every bootstrap sample.

The analyzer also enforces the gates rather than reporting around them: a cell
whose self-patch control is not exactly zero, whose embed arm moved, whose
ceiling arm is not 1.000, or whose tokenisation assertions failed is DROPPED and
the drop is counted and listed. Gates are never silently relaxed.

    python3 analyze_component_patch.py                      # both models, every stage present
    python3 analyze_component_patch.py --models qwen        # one model
    python3 analyze_component_patch.py --stage headline
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import patch_spec as M  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))
DEFAULT_ROOTS = {
    "qwen": os.environ.get(
        "EXP_M_QWEN_RESULTS",
        f"{REPO}/analysis/alfworld-qwen3.5-9b/mechanism/results/component-patching/qwen"),
    "gemma": os.environ.get(
        "EXP_M_GEMMA_RESULTS",
        f"{REPO}/analysis/alfworld-gemma-4-31b/mechanism"
        f"/results/component-patching/gemma"),
}
OUT_DIR = os.environ.get(
    "EXP_M_OUT",
    f"{REPO}/analysis/alfworld-qwen3.5-9b/mechanism/results/component-patching")
MODEL_LABEL = {"qwen": "Qwen3.5-9B", "gemma": "Gemma-4-31B-it"}
GEMMA_SOFT_DEFAULT = 280


# ══════════════════════════════════════════════════════════════════════════
# Loading
# ══════════════════════════════════════════════════════════════════════════

def cell_glob(arch, stage, modality, soft_tokens=GEMMA_SOFT_DEFAULT):
    root = DEFAULT_ROOTS[arch]
    if arch == "gemma":
        return os.path.join(root, f"soft{soft_tokens}", f"per_cell_{stage}",
                            modality, "*.json")
    return os.path.join(root, f"per_cell_{stage}", modality, "*.json")


def discover(arch, soft_tokens=GEMMA_SOFT_DEFAULT):
    """-> [(stage, modality, [paths])] for whatever has actually been run."""
    out = []
    for stage in ("headline", "bands", "all"):
        for modality in ("text", "visual"):
            paths = sorted(glob.glob(cell_glob(arch, stage, modality, soft_tokens)))
            if paths:
                out.append((stage, modality, paths))
    return out


def load_cells(paths):
    cells, bad = [], []
    for p in paths:
        try:
            with open(p, encoding="utf-8") as f:
                cells.append((p, json.load(f)))
        except Exception as e:                       # truncated write, corrupt json
            bad.append((p, f"{type(e).__name__}: {e}"))
    return cells, bad


# ══════════════════════════════════════════════════════════════════════════
# Per-arm aggregation
# ══════════════════════════════════════════════════════════════════════════

def analyze_arm(cells, arm_name, n_boot=M.N_BOOT, seed=M.SEED):
    """cells: [(path, rec)] that already passed the gate."""
    num_d, den_d, num_c, den_c = [], [], [], []
    deltas_d, deltas_c, dTs, moved, skipped = [], [], [], [], 0
    same_game_excluded = 0
    rand_partial = 0
    for _p, r in cells:
        a = r.get("arms", {}).get(arm_name)
        if a is None or a.get("skipped"):
            skipped += 1
            continue
        if arm_name.startswith("rand_") and a.get("rand_donor_game_mismatch"):
            # the donor came from the same game as the host: it may carry
            # game-specific exposure information, so it is not a valid
            # random-state control. Exclude it from the control (the headline
            # arms of this cell are unaffected and still counted) and say so.
            same_game_excluded += 1
            continue
        g = r["gap"]
        num_d.append((r["game_name"], a["delta_distractor"]))
        den_d.append((r["game_name"], g["distractor"]))
        num_c.append((r["game_name"], a["delta_correct"]))
        den_c.append((r["game_name"], g["correct"]))
        deltas_d.append(a["delta_distractor"])
        deltas_c.append(a["delta_correct"])
        dTs.append(r["delta_T"])
        moved.append(a.get("moved_distractor", 0.0))
        if a.get("rand_n_patched") is not None and a.get("rand_n_host"):
            if a["rand_n_patched"] < a["rand_n_host"]:
                rand_partial += 1

    if not num_d:
        return {"arm": arm_name, "n": 0, "skipped": skipped,
                "note": "no usable cells for this arm"}

    rec_d = M.clustered_ratio_ci(num_d, den_d, n_boot=n_boot, seed=seed)
    rec_c = M.clustered_ratio_ci(num_c, den_c, n_boot=n_boot, seed=seed)
    sign = M.sign_test(deltas_d)

    # delta_T provenance: the donor's activations were computed at absolute
    # positions shifted by delta_T. Correlate rather than assume it away.
    dt_corr = _pearson(dTs, deltas_d)

    out = {
        "arm": arm_name,
        "n": rec_d["n"], "n_games": rec_d["n_games"], "skipped": skipped,
        "recovered_distractor_pct": rec_d["ratio_pct"],
        "ci_lo_pct": rec_d["lo_pct"], "ci_hi_pct": rec_d["hi_pct"],
        "mean_R_nats": rec_d["mean_num"], "mean_G_nats": rec_d["mean_den"],
        "mean_R_ci": M.clustered_mean_ci(num_d, n_boot=n_boot, seed=seed),
        "mean_G_ci": M.clustered_mean_ci(den_d, n_boot=n_boot, seed=seed),
        "recovered_correct_pct": rec_c["ratio_pct"],
        "correct_ci_lo_pct": rec_c["lo_pct"], "correct_ci_hi_pct": rec_c["hi_pct"],
        "mean_R_correct_nats": rec_c["mean_num"], "mean_G_correct_nats": rec_c["mean_den"],
        "sign_test": sign,
        "delta_T_pearson_r": dt_corr[0], "delta_T_n": dt_corr[1],
        "mean_abs_moved": float(sum(abs(m) for m in moved) / len(moved)),
        "sites_nonempty_frac": float(sum(1 for m in moved if m > 0) / len(moved)),
    }
    if arm_name.startswith("rand_"):
        out["rand_same_game_excluded"] = same_game_excluded
        out["rand_partially_patched"] = rand_partial
    return out


def _pearson(xs, ys):
    import numpy as np
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 3 or x.std() == 0 or y.std() == 0:
        return float("nan"), int(x.size)
    return float(np.corrcoef(x, y)[0, 1]), int(x.size)


# ══════════════════════════════════════════════════════════════════════════
# Gates
# ══════════════════════════════════════════════════════════════════════════

def gate_report(loaded, bad_files):
    """-> (kept, report). A cell is kept only if every gate holds."""
    kept, dropped = [], []
    reasons = defaultdict(int)
    ceil_vals, self_vals, ref_vals = [], [], []
    for p, r in loaded:
        ok, why = M.cell_gate_ok(r)
        if ok:
            # numeric gates that cell_gate_ok only checked for presence
            ck = r["checks"]
            ceil_vals.append(ck.get("ceiling_recovery"))
            sp = ck.get("self_patch_delta", {})
            self_vals.append(max(abs(sp.get("distractor", 0.0)),
                                 abs(sp.get("correct", 0.0))))
            rc = ck.get("reference_consistency")
            if rc:
                ref_vals.append(max(rc[s]["abs_diff"] for s in ("distractor", "correct"))
                                if rc.get("ok") is not None else None)
            kept.append((p, r))
        else:
            dropped.append({"path": p, "state_id": r.get("state_id"),
                            "modality": r.get("modality"), "reasons": why})
            for w in why:
                reasons[w.split(" =")[0]] += 1
    for p, w in bad_files:
        dropped.append({"path": p, "reasons": [f"unreadable: {w}"]})
        reasons["unreadable"] += 1

    cv = [c for c in ceil_vals if c is not None]
    rv = [r for r in ref_vals if r is not None]
    report = {
        "n_kept": len(kept), "n_dropped": len(dropped),
        "drop_reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
        "dropped": dropped[:40],
        "ceiling_recovery_min": min(cv) if cv else None,
        "ceiling_recovery_max": max(cv) if cv else None,
        "self_patch_max_abs_delta": max(self_vals) if self_vals else None,
        "reference_max_abs_diff_nats": max(rv) if rv else None,
        "gates": {
            "embed_null": "every kept cell: delta == 0 and moved == 0",
            "self_patch": f"max |delta| = {max(self_vals) if self_vals else None} "
                          f"(tol {M.SELF_TOL})",
            "ceiling": f"recovery in [{min(cv) if cv else None}, "
                       f"{max(cv) if cv else None}] (tol |1-r| < {M.CEILING_TOL})",
            "reference": (f"max |diff| = {max(rv) if rv else None} nats "
                          f"(tol {M.REFERENCE_TOL})" if rv else "not run"),
        },
    }
    return kept, report


# ══════════════════════════════════════════════════════════════════════════
# One (model, stage, modality) block
# ══════════════════════════════════════════════════════════════════════════

def analyze_block(arch, stage, modality, paths, n_boot):
    loaded, bad = load_cells(paths)
    kept, gates = gate_report(loaded, bad)
    if not kept:
        return {"arch": arch, "stage": stage, "modality": modality,
                "gates": gates, "arms": {}, "note": "no cells passed the gates"}

    sample = kept[0][1]
    arms_spec = {a["name"]: a for a in M.build_arms(
        sample["num_layers"], sample["layer_types"], arch, stage)}
    order = [a["name"] for a in M.build_arms(
        sample["num_layers"], sample["layer_types"], arch, stage)]

    arms = {}
    for name in order:
        r = analyze_arm(kept, name, n_boot=n_boot)
        spec = arms_spec.get(name, {})
        r.update({"kind": spec.get("kind"), "component": spec.get("component"),
                  "scope": spec.get("scope", "both"), "donor": spec.get("donor")})
        for k in ("band", "layers", "depth_range"):
            if k in spec:
                r[k] = spec[k]
        arms[name] = r

    # exposure gap itself, for the record and for cross-model scale honesty
    gap_d = M.clustered_mean_ci([(r["game_name"], r["gap"]["distractor"])
                                 for _p, r in kept], n_boot=n_boot)
    gap_c = M.clustered_mean_ci([(r["game_name"], r["gap"]["correct"])
                                 for _p, r in kept], n_boot=n_boot)

    # per-state long table: the raw (G, R) pairs behind every arm. The headline
    # statistic is a ratio of means over these; the scatter figure draws the
    # pairs themselves so a reader can see the ceiling (points on y=x), the sign
    # of the MLP cloud, and the uncorrelated random-control cloud at once.
    per_state = []
    for _p, r in kept:
        for arm_name, a in r.get("arms", {}).items():
            if a.get("skipped"):
                continue
            per_state.append({
                "state_id": r["state_id"], "game": r["game_name"],
                "arm": arm_name,
                "G_d": r["gap"]["distractor"], "R_d": a["delta_distractor"],
                "G_c": r["gap"]["correct"], "R_c": a["delta_correct"],
                "delta_T": r["delta_T"],
            })

    return {
        "arch": arch, "model": MODEL_LABEL.get(arch, arch),
        "checkpoint": sample.get("provenance", {}).get("checkpoint"),
        "stage": stage, "modality": modality,
        "num_layers": sample["num_layers"],
        "depth_rule": sample.get("depth_rule"),
        "soft_tokens": sample.get("soft_tokens"),
        "hi_k": sample["hi_k"], "lo_k": sample["lo_k"],
        "provenance": sample.get("provenance"),
        "gates": gates,
        "exposure_gap_distractor_nats": gap_d,
        "exposure_gap_correct_nats": gap_c,
        "delta_T_mean": float(sum(r["delta_T"] for _p, r in kept) / len(kept)),
        "delta_T_range": [min(r["delta_T"] for _p, r in kept),
                          max(r["delta_T"] for _p, r in kept)],
        "n_action_tokens_mean": float(
            sum(r["n_action_tokens"]["distractor"] for _p, r in kept) / len(kept)),
        "arms": arms,
        "_per_state": per_state,
    }


# ══════════════════════════════════════════════════════════════════════════
# Cross-model reading
# ══════════════════════════════════════════════════════════════════════════

def cross_model(blocks):
    """The headline table: the SAME arm in both models. Only the recovered
    fraction is comparable — never the nats (different softcap, different vision
    token budget, sandwich norms on Gemma only)."""
    by_key = {}
    for b in blocks:
        for arm, r in b.get("arms", {}).items():
            if r.get("scope", "both") != "both":
                continue
            by_key.setdefault((b["stage"], b["modality"], arm), {})[b["arch"]] = r
    rows = []
    for (stage, modality, arm), per_arch in sorted(by_key.items()):
        if len(per_arch) < 2:
            continue
        q, g = per_arch["qwen"], per_arch["gemma"]
        rows.append({
            "stage": stage, "modality": modality, "arm": arm,
            "component": q.get("component"),
            "qwen_pct": q["recovered_distractor_pct"],
            "qwen_ci": [q["ci_lo_pct"], q["ci_hi_pct"]],
            "gemma_pct": g["recovered_distractor_pct"],
            "gemma_ci": [g["ci_lo_pct"], g["ci_hi_pct"]],
            "qwen_correct_pct": q["recovered_correct_pct"],
            "gemma_correct_pct": g["recovered_correct_pct"],
            "same_sign": (q["recovered_distractor_pct"]
                          * g["recovered_distractor_pct"]) > 0,
            "qwen_n": q["n"], "gemma_n": g["n"],
        })
    return rows


def verdicts(blocks):
    """The three control verdicts README §7 requires, stated per block."""
    out = []
    for b in blocks:
        arms = b.get("arms", {})
        if not arms:
            continue
        v = {"arch": b["arch"], "stage": b["stage"], "modality": b["modality"]}
        em = arms.get("embed", {})
        sp = arms.get("self_all", {})
        rl = arms.get("resid_last", {})
        v["embed_is_zero"] = abs(em.get("mean_R_nats") or 0.0) < M.SELF_TOL
        v["self_patch_is_zero"] = abs(sp.get("mean_R_nats") or 0.0) < M.SELF_TOL
        v["ceiling_is_one"] = abs((rl.get("recovered_distractor_pct") or 0.0) - 100.0) \
            < 100.0 * M.CEILING_TOL
        for nm in ("rand_all_mix", "rand_all_mlp"):
            r = arms.get(nm, {})
            pct = r.get("recovered_distractor_pct")
            lo, hi = r.get("ci_lo_pct"), r.get("ci_hi_pct")
            v[f"{nm}_pct"] = pct
            v[f"{nm}_ci_excludes_headline"] = None
            if pct is not None and lo is not None:
                head = arms.get("all_mix" if nm.endswith("mix") else "all_mlp", {})
                hp = head.get("recovered_distractor_pct")
                if hp is not None:
                    # the random control must not reproduce the headline effect
                    v[f"{nm}_ci_excludes_headline"] = bool(
                        not (lo <= hp <= hi) or abs(pct) < 0.25 * abs(hp))
            v[f"{nm}_same_game_excluded"] = r.get("rand_same_game_excluded")
            v[f"{nm}_n"] = r.get("n", 0)
        mx = arms.get("all_mix", {})
        ml = arms.get("all_mlp", {})
        v["mixer_pct"] = mx.get("recovered_distractor_pct")
        v["mlp_pct"] = ml.get("recovered_distractor_pct")
        v["ceiling_pct"] = rl.get("recovered_distractor_pct")
        # README §5.1: all_mix must land on the same ceiling as resid_last,
        # because the mixer is the only cross-row channel in both architectures.
        # If it does not, a hook is misfiring or the model has a cross-row path
        # we did not account for — either way the sweep is not measuring what
        # DESIGN says it measures.
        if v["mixer_pct"] is not None and v["ceiling_pct"] is not None:
            v["mixer_minus_ceiling_pts"] = v["mixer_pct"] - v["ceiling_pct"]
            v["mixer_at_ceiling"] = abs(v["mixer_minus_ceiling_pts"]) < 1.0
        # which depth range is necessary / sufficient
        for pref in ("mix", "mlp"):
            suff = [(b, arms.get(f"{pref}_band{b}", {}).get("recovered_distractor_pct"))
                    for b in range(M.N_BANDS)]
            nec = [(b, arms.get(f"{pref}_all_but_band{b}", {})
                    .get("recovered_distractor_pct")) for b in range(M.N_BANDS)]
            suff_v = [(b, p) for b, p in suff if p is not None and p == p]
            nec_v = [(b, p) for b, p in nec if p is not None and p == p]
            v[f"{pref}_most_sufficient_band"] = max(suff_v, key=lambda t: t[1])[0] if suff_v else None
            # the most NECESSARY band is the one whose removal costs the most,
            # i.e. the LOWEST all-but-band recovery
            v[f"{pref}_most_necessary_band"] = min(nec_v, key=lambda t: t[1])[0] if nec_v else None
            v[f"{pref}_band_sufficiency_pct"] = dict(suff)
            v[f"{pref}_band_necessity_pct"] = dict(nec)
        # The correct-action control is only MEANINGFUL on arms that are NOT at
        # the structural ceiling. `all_mix` reproduces the donor's entire score
        # for BOTH actions (mean_R_C == mean_G_C exactly, per state), so its
        # correct side can never discriminate PEB from arousal — reading it as a
        # control is the bug this replaces. all_mlp is below the ceiling, so
        # that is where distractor-specificity is testable, and it is tested in
        # NATS: a ratio against the tiny G_correct is unstable by construction.
        mx_c, mx_d = mx.get("mean_R_correct_nats"), mx.get("mean_R_nats")
        mx_gd, mx_gc = mx.get("mean_G_nats"), mx.get("mean_G_correct_nats")
        v["mixer_ceiling_signature"] = bool(
            mx_c is not None and mx_gc is not None and mx_d is not None
            and abs(mx_c - mx_gc) < 1e-6 and abs(mx_d - mx_gd) < 1e-6)
        mRc = ml.get("mean_R_correct_nats") or 0.0
        mRd = ml.get("mean_R_nats") or 0.0
        mGd = mx_gd or 1.0
        v["mlp_correct_over_distractor_nats"] = (abs(mRc) / abs(mRd)
                                                 if abs(mRd) > 1e-9 else None)
        v["mlp_peb_specific"] = (abs(mRc) < 0.25 * abs(mRd)
                                 if abs(mRd) > 1e-9 else None)
        if abs(mRd) < 0.10 * abs(mGd):
            v["verdict"] = ("no MLP effect to attribute (|R_D| < 10% of G); "
                            "PEB-specificity not testable on this block")
        elif v["mlp_peb_specific"]:
            v["verdict"] = "PEB-specific: the MLP effect is distractor-only"
        else:
            v["verdict"] = ("NOT PEB-specific: the MLP moves the correct action "
                            "with the distractor")
        out.append(v)
    return out


# ══════════════════════════════════════════════════════════════════════════
# Output
# ══════════════════════════════════════════════════════════════════════════

def write_csv(blocks, path):
    cols = ["arch", "model", "stage", "modality", "arm", "kind", "component",
            "scope", "donor", "band", "n", "n_games",
            "recovered_distractor_pct", "ci_lo_pct", "ci_hi_pct",
            "mean_R_nats", "mean_G_nats",
            "recovered_correct_pct", "correct_ci_lo_pct", "correct_ci_hi_pct",
            "sign_pos", "sign_neg", "sign_p",
            "delta_T_pearson_r", "mean_abs_moved", "sites_nonempty_frac"]
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for b in blocks:
            for arm, r in sorted(b.get("arms", {}).items()):
                st = r.get("sign_test") or {}
                w.writerow([b["arch"], b.get("model"), b["stage"], b["modality"],
                            arm, r.get("kind"), r.get("component"),
                            r.get("scope"), r.get("donor"), r.get("band"),
                            r.get("n"), r.get("n_games"),
                            r.get("recovered_distractor_pct"),
                            r.get("ci_lo_pct"), r.get("ci_hi_pct"),
                            r.get("mean_R_nats"), r.get("mean_G_nats"),
                            r.get("recovered_correct_pct"),
                            r.get("correct_ci_lo_pct"), r.get("correct_ci_hi_pct"),
                            st.get("pos"), st.get("neg"), st.get("p"),
                            r.get("delta_T_pearson_r"), r.get("mean_abs_moved"),
                            r.get("sites_nonempty_frac")])
    return path


def write_per_state_csv(blocks, path):
    import csv
    cols = ["arch", "modality", "stage", "state_id", "game", "arm",
            "G_d", "R_d", "G_c", "R_c", "delta_T"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for b in blocks:
            for row in b.get("_per_state", []):
                w.writerow([b["arch"], b["modality"], b["stage"],
                            row["state_id"], row["game"], row["arm"],
                            row["G_d"], row["R_d"], row["G_c"], row["R_c"],
                            row["delta_T"]])
    return path


def print_summary(blocks, xrows):
    print("\n" + "=" * 96)
    print("component-patching — Recovered exposure effect (mean(R)/mean(G), game-clustered 95% CI)")
    print("=" * 96)
    for b in blocks:
        g = b.get("gates", {})
        print(f"\n{b.get('model', b['arch'])} · {b['stage']} · {b['modality']} "
              f"· L={b.get('num_layers')} · k{b.get('hi_k')}→k{b.get('lo_k')}"
              + (f" · soft={b.get('soft_tokens')}" if b.get("soft_tokens") else ""))
        print(f"  cells kept {g.get('n_kept')} / dropped {g.get('n_dropped')}"
              + (f"  {g.get('drop_reasons')}" if g.get("n_dropped") else ""))
        gd = b.get("exposure_gap_distractor_nats") or {}
        print(f"  exposure gap G (distractor): {gd.get('point'):+.3f} nats "
              f"[{gd.get('lo'):+.3f},{gd.get('hi'):+.3f}]  n={gd.get('n')} "
              f"games={gd.get('n_games')}   delta_T mean {b.get('delta_T_mean'):+.1f} "
              f"range {b.get('delta_T_range')}")
        arms = b.get("arms", {})
        if not arms:
            print("  (no arms)")
            continue
        print(f"  {'arm':20s} {'kind':14s} {'recovered %':>26s} {'correct %':>10s} "
              f"{'sign +/-':>10s} {'n':>4s}")
        for name in ("embed", "self_all", "resid_last", "all_mix", "all_mlp",
                     "rand_all_mix", "rand_all_mlp",
                     "mix_band0", "mix_band1", "mix_band2", "mix_band3",
                     "mlp_band0", "mlp_band1", "mlp_band2", "mlp_band3",
                     "mix_all_but_band0", "mix_all_but_band1",
                     "mix_all_but_band2", "mix_all_but_band3",
                     "mlp_all_but_band0", "mlp_all_but_band1",
                     "mlp_all_but_band2", "mlp_all_but_band3",
                     "all_mix_full", "all_mix_lin",
                     "all_mix_postnorm", "all_mlp_postnorm"):
            r = arms.get(name)
            if not r or not r.get("n"):
                continue
            st = r.get("sign_test") or {}
            pct, lo, hi = (r["recovered_distractor_pct"], r["ci_lo_pct"],
                           r["ci_hi_pct"])
            s = (f"{pct:+8.1f} [{lo:+7.1f},{hi:+7.1f}]"
                 if pct == pct and lo == lo else f"{'nan':>26s}")
            c = r.get("recovered_correct_pct")
            cs = f"{c:+9.1f}" if c is not None and c == c else f"{'—':>10s}"
            print(f"  {name:20s} {str(r.get('kind')):14s} {s:>26s} {cs:>10s} "
                  f"{str(st.get('pos'))+'/'+str(st.get('neg')):>10s} {r['n']:>4d}")
    if xrows:
        print("\n" + "-" * 96)
        print("CROSS-MODEL (same arm, same definition — only the ratio is comparable)")
        print("-" * 96)
        print(f"  {'stage/modality/arm':34s} {'Qwen3.5-9B':>22s} {'Gemma-4-31B':>22s} "
              f"{'same sign':>10s}")
        for r in xrows:
            q = f"{r['qwen_pct']:+7.1f} [{r['qwen_ci'][0]:+6.1f},{r['qwen_ci'][1]:+6.1f}]"
            g = f"{r['gemma_pct']:+7.1f} [{r['gemma_ci'][0]:+6.1f},{r['gemma_ci'][1]:+6.1f}]"
            print(f"  {r['stage']+'/'+r['modality']+'/'+r['arm']:34s} {q:>22s} "
                  f"{g:>22s} {('yes' if r['same_sign'] else 'NO'):>10s}")
    print("=" * 96)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["qwen", "gemma"],
                    choices=["qwen", "gemma"])
    ap.add_argument("--stage", nargs="*", default=None,
                    choices=["headline", "bands", "all"])
    ap.add_argument("--soft-tokens", type=int, default=GEMMA_SOFT_DEFAULT)
    ap.add_argument("--n-boot", type=int, default=M.N_BOOT)
    ap.add_argument("--out", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    blocks = []
    for arch in args.models:
        found = discover(arch, args.soft_tokens)
        if not found:
            print(f"[M] no component-patching cells found for {arch} under {DEFAULT_ROOTS[arch]}")
            continue
        for stage, modality, paths in found:
            if args.stage and stage not in args.stage:
                continue
            print(f"[M] analyzing {arch}/{stage}/{modality}: {len(paths)} cells")
            blocks.append(analyze_block(arch, stage, modality, paths, args.n_boot))

    if not blocks:
        print("[M] nothing to analyze — run a runner first")
        raise SystemExit(1)

    xrows = cross_model(blocks)
    verdict_rows = verdicts(blocks)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_json = args.out or os.path.join(OUT_DIR, "analysis.json")
    payload = {
        "metric": "recovered exposure effect = 100 * mean(R)/mean(G), "
                  "game-clustered bootstrap, ratio recomputed per sample",
        "n_boot": args.n_boot, "seed": M.SEED,
        "exposure": {"hi_k": M.K_HI, "lo_k": M.K_LO},
        "depth_rule": "d = l/(L-1); bands fixed a priori at [0,.25,.5,.75,1]",
        "comparability": "ratios only; absolute nats are NOT comparable across "
                         "models (softcap, vision token budget, sandwich norms)",
        "blocks": blocks,
        "cross_model": xrows,
        "control_verdicts": verdict_rows,
    }
    tmp = out_json + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)
    os.replace(tmp, out_json)
    csv_path = write_csv(blocks, os.path.join(OUT_DIR, "summary.csv"))
    ps_path = write_per_state_csv(blocks, os.path.join(OUT_DIR, "per_state.csv"))
    for b in blocks:                       # keep the JSON lean
        b.pop("_per_state", None)
    print(f"\n[M] wrote {out_json}\n[M] wrote {csv_path}")
    if not args.quiet:
        print_summary(blocks, xrows)

    # hard gate failures are surfaced as a non-zero exit so a launch script
    # cannot proceed to plotting on a broken run
    bad = []
    for v in verdict_rows:
        for k in ("embed_is_zero", "self_patch_is_zero", "ceiling_is_one"):
            if v.get(k) is False:
                bad.append(f"{v['arch']}/{v['stage']}/{v['modality']}: {k}")
        # a couple of same-game random donors in a block is a provenance wart,
        # handled by exclusion; a block where the control is mostly same-game is
        # a broken control and must stop the pipeline.
        for nm in ("rand_all_mix", "rand_all_mlp"):
            exc = v.get(f"{nm}_same_game_excluded") or 0
            n = (v.get(f"{nm}_n") or 0) + exc
            # absolute floor, not a bare fraction: a half-finished block has a
            # small denominator and one excluded cell would trip a 2% rule while
            # the run is still healthy. Exclusion already protects the control;
            # this only stops the pipeline when the control is broadly broken.
            if exc >= 3 or (n and exc / n > 0.05):
                bad.append(f"{v['arch']}/{v['stage']}/{v['modality']}: "
                           f"{nm} has {exc}/{n} same-game donors")
    if bad:
        print("\n[M] GATE FAILURES:\n  " + "\n  ".join(bad))
        raise SystemExit(2)


if __name__ == "__main__":
    main()
