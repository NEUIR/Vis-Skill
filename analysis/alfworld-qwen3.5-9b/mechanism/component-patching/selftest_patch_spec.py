#!/usr/bin/env python3
"""
CPU-only self-test for the component-patching shared spec.

Runs without a GPU, without transformers, and without loading any weights: it
reads only the two checkpoints' `config.json` files for their `layer_types`.
Everything here must pass before either runner is launched — the point is that a
definition bug is caught in 2 seconds on CPU instead of 5 hours into a 31B sweep.

    python3 selftest_patch_spec.py
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import patch_spec as M  # noqa: E402

QWEN_CONFIG = os.environ.get("QWEN_CONFIG", os.path.join(os.environ.get("MODELS_DIR", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "models"))), "Qwen3.5-9B", "config.json"))
GEMMA_CONFIG = os.environ.get("GEMMA_CONFIG", os.path.join(os.environ.get("MODELS_DIR", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "models"))), "Gemma-4-31B-it", "config.json"))

FAILS: list = []


def check(name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    print(f"  [{tag}] {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def layer_types(path):
    with open(path) as f:
        c = json.load(f)
    tc = c.get("text_config", c)
    return tc["layer_types"], tc["num_hidden_layers"]


def main():
    print("=" * 78)
    print("component-patching spec self-test (CPU only)")
    print("=" * 78)

    # ── 1. depth bands ────────────────────────────────────────────────────
    print("\n1. depth bands are fixed, exhaustive and identical in rule")
    b32 = M.depth_bands(32)
    b60 = M.depth_bands(60)
    check("L=32 bands", b32 == [list(range(0, 8)), list(range(8, 16)),
                                list(range(16, 24)), list(range(24, 32))],
          f"{[len(b) for b in b32]} layers/band")
    check("L=60 bands", b60 == [list(range(0, 15)), list(range(15, 30)),
                                list(range(30, 45)), list(range(45, 60))],
          f"{[len(b) for b in b60]} layers/band")
    check("last layer in the top band (L=32)", 31 in b32[-1])
    check("last layer in the top band (L=60)", 59 in b60[-1])
    check("d(0)=0 and d(L-1)=1 for both", M.depth_frac(0, 32) == 0.0
          and M.depth_frac(31, 32) == 1.0
          and M.depth_frac(0, 60) == 0.0 and M.depth_frac(59, 60) == 1.0)
    # the bands must not depend on the model's layer count in *relative* terms
    for L in (32, 60):
        for b, layers in enumerate(M.depth_bands(L)):
            ds = [M.depth_frac(l, L) for l in layers]
            lo, hi = M.BAND_EDGES[b], M.BAND_EDGES[b + 1]
            ok = all(lo - 1e-12 <= d <= hi + 1e-12 for d in ds)
            check(f"L={L} band{b} inside [{lo},{hi}]", ok,
                  f"d in [{min(ds):.3f},{max(ds):.3f}]")

    # ── 2. arms ───────────────────────────────────────────────────────────
    print("\n2. arm sets")
    lt_q, L_q = layer_types(QWEN_CONFIG)
    lt_g, L_g = layer_types(GEMMA_CONFIG)
    print(f"     Qwen: L={L_q}, {lt_q.count('linear_attention')} linear / "
          f"{lt_q.count('full_attention')} full")
    print(f"     Gemma: L={L_g}, {lt_g.count('sliding_attention')} sliding / "
          f"{lt_g.count('full_attention')} full")
    check("Qwen layer_types vocabulary", set(lt_q) <= {"linear_attention", "full_attention"},
          str(sorted(set(lt_q))))
    check("Gemma layer_types vocabulary", set(lt_g) <= {"sliding_attention", "full_attention"},
          str(sorted(set(lt_g))))
    check("Qwen is 32 layers", L_q == 32, str(L_q))
    check("Gemma is 60 layers", L_g == 60, str(L_g))

    aq = M.build_arms(L_q, lt_q, "qwen", "all")
    ag = M.build_arms(L_g, lt_g, "gemma", "all")
    nq = {a["name"] for a in aq}
    ng = {a["name"] for a in ag}

    HEADLINE = {"all_mix", "all_mlp"}
    CONTROL = {"embed", "self_all", "rand_all_mix", "rand_all_mlp"}
    ANCHOR = {"resid_last"}
    BANDS = {f"{c}_band{b}" for c in ("mix", "mlp") for b in range(M.N_BANDS)}

    for label, names, need in (("Qwen", nq, HEADLINE | CONTROL | ANCHOR | BANDS),
                               ("Gemma", ng, HEADLINE | CONTROL | ANCHOR | BANDS)):
        check(f"{label} has every shared arm", need <= names,
              f"missing {sorted(need - names)}")
    check("headline arms are byte-identical names across models",
          HEADLINE <= (nq & ng))
    check("Qwen-only arms absent from Gemma",
          {"all_mix_full", "all_mix_lin"} <= nq and not ({"all_mix_full", "all_mix_lin"} & ng))
    check("Gemma-only arms absent from Qwen",
          {"all_mix_postnorm", "all_mlp_postnorm"} <= ng
          and not ({"all_mix_postnorm", "all_mlp_postnorm"} & nq))

    by_name_q = {a["name"]: a for a in aq}
    by_name_g = {a["name"]: a for a in ag}
    check("Qwen all_mix covers all 32 mixers",
          by_name_q["all_mix"]["sites"] == [M.mix_site(l) for l in range(32)])
    check("Gemma all_mix covers all 60 mixers",
          by_name_g["all_mix"]["sites"] == [M.mix_site(l) for l in range(60)])
    check("Qwen all_mlp covers all 32 MLPs",
          by_name_q["all_mlp"]["sites"] == [M.mlp_site(l) for l in range(32)])
    check("Gemma all_mlp covers all 60 MLPs",
          by_name_g["all_mlp"]["sites"] == [M.mlp_site(l) for l in range(60)])
    check("Qwen full+lin partition the mixers",
          set(by_name_q["all_mix_full"]["sites"]) | set(by_name_q["all_mix_lin"]["sites"])
          == set(by_name_q["all_mix"]["sites"])
          and not (set(by_name_q["all_mix_full"]["sites"])
                   & set(by_name_q["all_mix_lin"]["sites"])))
    check("Qwen full-attn count == 8", len(by_name_q["all_mix_full"]["layers"]) == 8,
          str(by_name_q["all_mix_full"]["layers"]))
    check("Qwen GDN count == 24", len(by_name_q["all_mix_lin"]["layers"]) == 24)
    for arch, by_name, L in (("Qwen", by_name_q, 32), ("Gemma", by_name_g, 60)):
        union, inter = set(), set()
        for b in range(M.N_BANDS):
            s = set(by_name[f"mix_band{b}"]["sites"])
            inter |= (union & s)
            union |= s
        check(f"{arch} mix bands partition all {L} mixers",
              union == set(by_name["all_mix"]["sites"]) and not inter)
        union, inter = set(), set()
        for b in range(M.N_BANDS):
            s = set(by_name[f"mlp_band{b}"]["sites"])
            inter |= (union & s)
            union |= s
        check(f"{arch} mlp bands partition all {L} MLPs",
              union == set(by_name["all_mlp"]["sites"]) and not inter)
    check("self_all patches every headline site (strong harness control)",
          set(by_name_q["self_all"]["sites"])
          == set(by_name_q["all_mix"]["sites"]) | set(by_name_q["all_mlp"]["sites"]))
    check("donors are as specified",
          by_name_q["self_all"]["donor"] == "self"
          and by_name_q["rand_all_mix"]["donor"] == "rand"
          and by_name_q["all_mix"]["donor"] == "hi"
          and by_name_q["resid_last"]["donor"] == "hi")

    # leave-one-band-out arms must be the exact complement of the band arms:
    # sufficiency (band alone) and necessity (all but band) have to partition the
    # same layer set, or the two rows of the Appendix figure are not comparable.
    for label, by_name, L in (("Qwen", by_name_q, 32), ("Gemma", by_name_g, 60)):
        for comp, pref in (("token_mixer", "mix"), ("mlp", "mlp")):
            allsites = set(by_name[f"all_{pref}"]["sites"])
            ok_comp = ok_disj = True
            for b in range(M.N_BANDS):
                only = set(by_name[f"{pref}_band{b}"]["sites"])
                loo = set(by_name[f"{pref}_all_but_band{b}"]["sites"])
                ok_comp &= (only | loo) == allsites
                ok_disj &= not (only & loo)
            check(f"{label} {comp}: band ∪ all-but-band == all layers", ok_comp)
            check(f"{label} {comp}: band ∩ all-but-band == ∅", ok_disj)
        check(f"{label} LOO arms are marked band_loo",
              all(by_name[f"mix_all_but_band{b}"]["kind"] == "band_loo"
                  for b in range(M.N_BANDS)))
    check("stage=headline excludes band arms",
          not any(a["kind"] in ("band", "band_loo")
                  for a in M.build_arms(L_q, lt_q, "qwen", "headline")))
    check("stage=headline excludes arch-specific arms",
          not any(a["kind"] == "arch_specific"
                  for a in M.build_arms(L_q, lt_q, "qwen", "headline"))
          and not any(a["kind"] == "arch_specific"
                      for a in M.build_arms(L_g, lt_g, "gemma", "headline")))

    # ── 3. random-donor map ───────────────────────────────────────────────
    print("\n3. random-donor map is deterministic and game-disjoint")
    entries = [{"state_id": f"g{g}__s{s}", "game_name": f"game{g}"}
               for g in range(64) for s in range(2, 3)]
    entries = entries[:165]
    order1 = M.build_donor_order(entries, M.SEED)
    order2 = M.build_donor_order(entries, M.SEED)
    check("deterministic for a fixed seed", order1 == order2)
    check("order is a permutation of the states",
          sorted(order1) == sorted(e["state_id"] for e in entries))
    game_of = {e["state_id"]: e["game_name"] for e in entries}
    bad = [(order1[i - 1], order1[i]) for i in range(1, len(order1))
           if game_of[order1[i - 1]] == game_of[order1[i]]]
    check("no adjacent pair shares a game", not bad, f"{len(bad)} violations")
    check("first state has no donor (documented, not a bug)",
          M.donor_partner(order1, 0) == (None, False))
    check("state i's donor is state i-1",
          M.donor_partner(order1, 7)[0] == order1[6])
    # adversarial: one game holding most states has NO valid game-disjoint order.
    # The builder must degrade to a full permutation (with adjacent same-game
    # pairs) rather than loop forever, and the runner must be able to see it.
    lopsided = ([{"state_id": f"big__s{i}", "game_name": "biggame"} for i in range(20)]
                + [{"state_id": f"small__s{i}", "game_name": f"sg{i}"} for i in range(4)])
    ordl = M.build_donor_order(lopsided, M.SEED)
    gofl = {e["state_id"]: e["game_name"] for e in lopsided}
    viol = sum(1 for i in range(1, len(ordl))
               if gofl[ordl[i - 1]] == gofl[ordl[i]])
    check("lopsided input still yields a full permutation",
          sorted(ordl) == sorted(e["state_id"] for e in lopsided))
    check("lopsided input terminates with flagged same-game adjacencies",
          viol > 0, f"{viol} adjacent same-game pairs (runner must record these)")

    # ── 4. metric: ratio of means, never mean of ratios ───────────────────
    print("\n4. metric")
    # R = 0.5 G exactly for every state -> 50% by either estimator
    num = [(f"g{i % 8}", 0.5 * (i + 1)) for i in range(40)]
    den = [(f"g{i % 8}", float(i + 1)) for i in range(40)]
    r = M.clustered_ratio_ci(num, den, n_boot=500)
    check("exact 50% recovery is reported as 50%", abs(r["ratio_pct"] - 50.0) < 1e-9,
          f"{r['ratio_pct']:.6f}")
    # a state with a tiny G must NOT blow the estimate up: ratio-of-means vs
    # mean-of-ratios differ hugely here, and we must be on the ratio-of-means side
    num2 = [("gA", 1.0), ("gA", 1.0), ("gB", 0.5), ("gB", 0.5)]
    den2 = [("gA", 2.0), ("gA", 2.0), ("gB", 0.001), ("gB", 0.001)]
    r2 = M.clustered_ratio_ci(num2, den2, n_boot=500)
    # game means: num gA=1.0 gB=0.5 -> 0.75 ; den gA=2.0 gB=0.001 -> 1.0005
    expect2 = 100.0 * 0.75 / 1.0005
    mean_of_ratios = 100.0 * sum(n / d for (_, n), (_, d) in zip(num2, den2)) / 4
    check("near-zero denominator does not explode the estimate",
          abs(r2["ratio_pct"] - expect2) < 1e-9,
          f"ratio-of-means {r2['ratio_pct']:.4f}% (expected {expect2:.4f}%) vs "
          f"mean-of-ratios {mean_of_ratios:.0f}%")
    check("mean-of-ratios would have been absurd", mean_of_ratios > 10000,
          f"{mean_of_ratios:.0f}%")

    # clustering must WIDEN the CI relative to a state-level bootstrap: 165
    # states come from only 64 games, and README §6 says a state-level bootstrap
    # would understate the CI. Fixture: 8 games x 10 states, a large between-game
    # spread and no within-game spread, so the only variance is the clustering.
    rows = [(f"g{g}", float(g)) for g in range(8) for _ in range(10)]
    cm = M.clustered_mean_ci(rows, n_boot=2000)
    import numpy as np
    vals = np.asarray([v for _, v in rows], dtype=np.float64)
    rng = np.random.default_rng(M.SEED)
    flat = vals[rng.integers(0, vals.size, size=(2000, vals.size))].mean(axis=1)
    lo_s, hi_s = np.quantile(flat, [0.025, 0.975])
    check("clustered mean CI brackets the point",
          cm["lo"] <= cm["point"] <= cm["hi"],
          f"{cm['point']:.3f} [{cm['lo']:.3f},{cm['hi']:.3f}]")
    check("clustering widens the CI vs a state-level bootstrap",
          (cm["hi"] - cm["lo"]) > (hi_s - lo_s),
          f"clustered width {cm['hi']-cm['lo']:.3f} vs state-level {hi_s-lo_s:.3f}")
    # the widening factor is not arbitrary: with 10 states per game and no
    # within-game variance, the clustered bootstrap draws 8 game means where the
    # state-level one draws 80 states, so the width ratio must be
    # sqrt(n_states / n_games). Asserting the theory catches an implementation
    # that clusters by the wrong key (which would silently return the flat width).
    theory = (len(rows) / cm["n_games"]) ** 0.5
    observed = (cm["hi"] - cm["lo"]) / max(hi_s - lo_s, 1e-12)
    check("widening factor matches sqrt(n_states/n_games)",
          abs(observed / theory - 1.0) < 0.25,
          f"observed {observed:.3f} vs theory {theory:.3f}")
    check("n_games is reported", cm["n_games"] == 8, str(cm["n_games"]))
    check("n states is reported", cm["n"] == 80, str(cm["n"]))
    try:
        M.clustered_ratio_ci([("a", 1.0)], [("b", 1.0), ("b", 2.0)])
        check("mismatched num/den lengths raise", False)
    except ValueError:
        check("mismatched num/den lengths raise", True)
    try:
        M.clustered_ratio_ci([("a", 1.0), ("b", 1.0)], [("b", 1.0), ("a", 1.0)])
        check("misaligned games raise", False)
    except ValueError:
        check("misaligned games raise", True)
    st = M.sign_test([1.0, 2.0, -1.0])
    check("sign_test counts", st["pos"] == 2 and st["neg"] == 1, str(st))

    # ── 5. cell gate ──────────────────────────────────────────────────────
    print("\n5. cell gate")
    good = {f: 1 for f in M.REQUIRED_CELL_FIELDS}
    good.update({"checks": {c: True for c in M.REQUIRED_CHECKS},
                 "arms": {n: {} for n in M.GATE_ARMS}})
    ok, why = M.cell_gate_ok(good)
    check("a complete passing cell is accepted", ok, str(why))
    bad = json.loads(json.dumps(good))
    bad["checks"]["self_patch_ok"] = False
    ok2, why2 = M.cell_gate_ok(bad)
    check("a failed self-patch control rejects the cell", not ok2, str(why2))
    bad2 = json.loads(json.dumps(good))
    del bad2["arms"]["resid_last"]
    ok3, why3 = M.cell_gate_ok(bad2)
    check("a missing ceiling anchor rejects the cell", not ok3, str(why3))
    bad3 = json.loads(json.dumps(good))
    del bad3["delta_T"]
    ok4, why4 = M.cell_gate_ok(bad3)
    check("a cell without delta_T provenance is rejected", not ok4, str(why4))

    print("\n" + "=" * 78)
    if FAILS:
        print(f"SELF-TEST FAILED — {len(FAILS)} checks: {FAILS}")
        print("=" * 78)
        raise SystemExit(1)
    print("SELF-TEST PASSED — spec is internally consistent for L=32 and L=60")
    print("=" * 78)


if __name__ == "__main__":
    main()
