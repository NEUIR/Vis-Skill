#!/usr/bin/env python3
"""
CPU-only end-to-end test of the component-patching patch harness, on a toy model.

The real runs cost ~5 GPU-hours per model per stage, and every one of their
numbers rests on three properties that must hold EXACTLY, not approximately:

  * the `embed` arm moves nothing (the action tokens are identical by
    construction between k=0 and k=8);
  * the `self_all` arm changes nothing (the harness reproduces its own forward);
  * the `resid_last` arm recovers exactly 100% of the gap (the final residual
    determines the logits).

If any of those is off, the intervention is not doing what README.md says and the
recovered fractions are meaningless. This test builds a small row-mixing model,
runs the real `patch_harness.run_cell` against it — including the case where the
donor and host prompts have DIFFERENT lengths, which is the whole reason
alignment is by action-token index — and then feeds the resulting record through
the real analyzer.

Nothing here needs a GPU or a checkpoint.

    python3 selftest_patch_harness.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import patch_harness as H  # noqa: E402
import patch_spec as M     # noqa: E402

FAILS: list = []
L = 8
D = 16
V = 64
LAYER_TYPES = ["linear_attention", "full_attention"] * (L // 2)


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


# ══════════════════════════════════════════════════════════════════════════
# A toy model with the same module topology the harness hooks
# ══════════════════════════════════════════════════════════════════════════

class FakeMixer(nn.Module):
    """Stands in for attention / GatedDeltaNet. Returns a TUPLE, like
    Qwen3_5Attention and Gemma4TextAttention do, so the takes_tuple path is
    exercised. The causal mean pool makes it genuinely row-mixing: patching one
    row's mixer output does NOT reproduce the donor's downstream computation,
    because later layers re-pool over rows we did not patch."""

    def __init__(self, d):
        super().__init__()
        self.proj = nn.Linear(d, d)

    def forward(self, x):
        c = torch.cumsum(x, dim=1)
        n = torch.arange(1, x.shape[1] + 1, device=x.device, dtype=x.dtype)[:, None]
        return self.proj(x + c / n), None


class FakeMLP(nn.Module):
    """Returns a BARE tensor, like Qwen3_5MLP and Gemma4TextMLP."""

    def __init__(self, d):
        super().__init__()
        self.fc = nn.Linear(d, d)

    def forward(self, x):
        return torch.tanh(self.fc(x))


class FakeLayer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.mixer = FakeMixer(d)
        self.mlp = FakeMLP(d)

    def forward(self, x):
        h, _ = self.mixer(x)
        x = x + h
        return x + self.mlp(x)


class FakeModel(nn.Module):
    def __init__(self, d=D, v=V, layers=L):
        super().__init__()
        self.embed_tokens = nn.Embedding(v, d)
        self.layers = nn.ModuleList([FakeLayer(d) for _ in range(layers)])
        self.head = nn.Linear(d, v, bias=False)

    def forward(self, input_ids, **kw):
        x = self.embed_tokens(input_ids)
        for l in self.layers:
            x = l(x)
        return self.head(x)


# ══════════════════════════════════════════════════════════════════════════
# Toy adapter: the same interface the two real runners implement
# ══════════════════════════════════════════════════════════════════════════

class FakeAdapter:
    arch = "qwen"                 # reuse the Qwen arm set, incl. full/GDN split
    model_name = "FakeModel"
    num_layers = L
    layer_types = LAYER_TYPES

    def __init__(self, model, vocab):
        self.model = model
        self.vocab = vocab        # token id lookup for fake action strings
        self.device = "cpu"
        self.states = {}

    def load_state(self, sid):
        return self.states[sid]

    def build_cell(self, state, modality, k):
        """k=8 gets a LONGER document than k=0, so delta_T != 0 and the test
        genuinely exercises index-based rather than position-based alignment."""
        doc = state["doc_k8"] if k == M.K_HI else state["doc_k0"]
        stem = state["stem"]
        out = {"action_start": len(doc) + len(stem), "soft_tokens": None,
               "n_image_tokens": None, "card_paths": None, "images": None}
        for side in ("correct", "distractor"):
            act = state[f"{side}_ids"]
            ids = doc + stem + act
            out[f"forced_{side}_ids"] = ids
            out[f"stem_prefix_ok_{side[0]}"] = (
                ids[:out["action_start"]] == doc + stem)
        return out

    def inputs(self, cell, side, ids):
        return {"input_ids": torch.tensor([list(ids)], dtype=torch.long)}

    def forward(self, inputs):
        with torch.no_grad():
            return self.model(**inputs)

    def reference_score(self, cell, state):
        out = {}
        for side in ("correct", "distractor"):
            ids = cell[f"forced_{side}_ids"]
            rows, targets = H.rows_and_targets(ids, cell["action_start"])
            with torch.no_grad():
                logits = self.model(input_ids=torch.tensor([ids]))
            lp, _ = H.action_logp(logits, rows, targets)
            out[f"logp_{side}"] = lp
        return out

    def release(self, cell, side):
        return None

    def release_all(self, cell):
        return None

    def sites(self, wanted):
        all_sites = {M.EMBED_SITE: (self.model.embed_tokens, False)}
        for i, layer in enumerate(self.model.layers):
            all_sites[M.mix_site(i)] = (layer.mixer, True)    # tuple-returning
            all_sites[M.mlp_site(i)] = (layer.mlp, False)     # bare tensor
            all_sites[M.resid_site(i)] = (layer, False)
        missing = [w for w in wanted if w not in all_sites]
        if missing:
            raise RuntimeError(f"unknown sites {missing}")
        return {w: all_sites[w] for w in wanted}

    def provenance(self):
        return {"checkpoint": "toy", "softcap": None, "forward": "single forward"}


# ══════════════════════════════════════════════════════════════════════════

def make_state(rng, sid="toy__step1", game="toygame"):
    """A fake decision state: two documents of DIFFERENT length, a shared stem,
    and two forced actions with different token counts."""
    def ids(n, lo=5):
        return [int(rng.integers(lo, V)) for _ in range(n)]

    stem = ids(6)
    return {
        "state_id": sid, "game_name": game, "task_type": "toy",
        "distractor_keyword": "toykw",
        "doc_k0": ids(11), "doc_k8": ids(19),      # k=8 is 8 tokens longer
        "stem": stem,
        "correct_ids": ids(5), "distractor_ids": ids(3),
        "correct_action": "toy correct", "distractor_action": "toy distractor",
    }


def main():
    torch.manual_seed(0)
    print("=" * 78)
    print("component-patching harness self-test (toy model, CPU)")
    print("=" * 78)

    model = FakeModel().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    adapter = FakeAdapter(model, V)
    rng = torch.Generator().manual_seed(7)
    import numpy as np
    nprng = np.random.default_rng(7)

    s_a = make_state(nprng, "toyA__step1", "gameA")
    s_b = make_state(nprng, "toyB__step1", "gameB")

    arms = M.build_arms(L, LAYER_TYPES, "qwen", "all")
    wanted = M.required_sites(arms, L, "qwen")
    patcher = H.RowPatcher(adapter.sites(wanted)).install()
    check("every arm site resolves to a real module", len(patcher.sites) == len(wanted),
          f"{len(patcher.sites)} sites, {len(arms)} arms")

    # ── 1. the fixture really is length-misaligned ────────────────────────
    print("\n1. fixture exercises length misalignment")
    hi = adapter.build_cell(s_a, "text", M.K_HI)
    lo = adapter.build_cell(s_a, "text", M.K_LO)
    dT = hi["action_start"] - lo["action_start"]
    check("delta_T != 0 (donor and host prompts differ in length)", dT != 0,
          f"delta_T = {dT}")
    check("action token ids are identical across exposures",
          all(hi[f"forced_{s}_ids"][hi["action_start"]:]
              == lo[f"forced_{s}_ids"][lo["action_start"]:]
              for s in ("correct", "distractor")))
    check("the two forced actions have different token counts",
          len(s_a["correct_ids"]) != len(s_a["distractor_ids"]),
          f"{len(s_a['correct_ids'])} vs {len(s_a['distractor_ids'])}")
    rows_lo, tgt_lo = H.rows_and_targets(lo["forced_distractor_ids"],
                                         lo["action_start"])
    check("rows are exactly the rows predicting the action tokens",
          len(rows_lo) == len(s_a["distractor_ids"])
          and rows_lo[0] == lo["action_start"] - 1,
          f"rows {rows_lo}")

    # ── 2. the real run_cell, with a random-state donor from another game ──
    print("\n2. run_cell gates (must be EXACT)")
    donor_b = {}
    cell_b = adapter.build_cell(s_b, "text", M.K_HI)
    for side in ("distractor", "correct"):
        d = H._donor_pass(adapter, patcher, cell_b, side)
        donor_b[side] = d["store"]
    rand = H.donor_keep(donor_b, s_b)

    rec, tagged = H.run_cell(adapter, patcher, s_a, "text", arms, rand, verify=True)
    ck = rec["checks"]
    check("cell_gate_ok accepts the record", M.cell_gate_ok(rec)[0],
          str(M.cell_gate_ok(rec)[1]))
    check("embed arm moved nothing", ck["embed_null_ok"] is True)
    check("embed delta is exactly zero",
          rec["arms"]["embed"]["delta_distractor"] == 0.0
          and rec["arms"]["embed"]["moved_distractor"] == 0.0,
          f"delta={rec['arms']['embed']['delta_distractor']} "
          f"moved={rec['arms']['embed']['moved_distractor']}")
    check("self_all delta is exactly zero on both actions",
          ck["self_patch_ok"] is True
          and rec["arms"]["self_all"]["delta_distractor"] == 0.0
          and rec["arms"]["self_all"]["delta_correct"] == 0.0,
          f"{ck['self_patch_delta']}")
    check("resid_last recovers exactly 100% of the gap",
          ck["ceiling_ok"] is True
          and abs(rec["arms"]["resid_last"]["recovery_distractor"] - 1.0) < 1e-6,
          f"recovery={rec['arms']['resid_last']['recovery_distractor']:.9f}")
    check("reference scorer agrees with the harness baseline",
          ck["reference_consistency"]["ok"] is True,
          f"max |diff| = {max(v['abs_diff'] for k, v in ck['reference_consistency'].items() if isinstance(v, dict)):.2e}")
    check("every site was recorded in both passes", ck["all_sites_recorded"] is True)
    check("delta_T is recorded as provenance", rec["delta_T"] == dT)
    check("the returned donor is tagged for the next cell's random control",
          tagged.get("_state_id") == s_a["state_id"] and "distractor" in tagged)

    # ── 3. the arms are non-trivial and donor-specific ────────────────────
    print("\n3. arms are non-trivial and donor-specific")
    A = rec["arms"]
    check("the exposure gap is non-zero (otherwise nothing to recover)",
          abs(rec["gap"]["distractor"]) > 1e-6,
          f"G = {rec['gap']['distractor']:+.6f} nats")
    check("all_mix actually moved activations",
          A["all_mix"]["moved_distractor"] > 0, f"{A['all_mix']['moved_distractor']:.3f}")
    check("all_mix changed the score",
          abs(A["all_mix"]["delta_distractor"]) > 1e-9,
          f"delta={A['all_mix']['delta_distractor']:+.6f} "
          f"({100*A['all_mix']['recovery_distractor']:+.1f}% of gap)")
    check("all_mlp changed the score",
          abs(A["all_mlp"]["delta_distractor"]) > 1e-9,
          f"delta={A['all_mlp']['delta_distractor']:+.6f}")
    # STRUCTURAL CEILING, not a bug — and the reason is worth stating precisely.
    # Both real models are row-wise everywhere EXCEPT inside the mixer module:
    # the residual add, RMSNorm (including Gemma's sandwich norms) and the MLP
    # (Linear -> act -> Linear) all act per row. So at a decision row r, if
    #   (i)  the embedding at r is identical between donor and host — guaranteed
    #        by `action_ids_match`, the same assertion that licenses index
    #        alignment, and verified by the `embed` arm being exactly zero, and
    #   (ii) every mixer output at r is replaced by the donor's,
    # then by induction the residual at r matches the donor's after every layer,
    # and `all_mix` reproduces S_{k=8} exactly. Overwriting the mixer output
    # AFTER the cross-row mixing has happened is precisely what makes this exact;
    # the unpatched document rows can no longer influence row r.
    # Consequence for the paper: `all_mix ~= 100%` is a harness ceiling shared
    # with `resid_last`, NOT evidence about exposure. The informative arms are
    # the ones that patch a SUBSET (the depth bands) and the MLP, whose value is
    # not structurally pinned.
    ceil = A["resid_last"]["recovery_distractor"]
    check("all_mix hits the structural ceiling (== resid_last)",
          abs(A["all_mix"]["recovery_distractor"] - ceil) < 1e-6,
          f"all_mix={A['all_mix']['recovery_distractor']:.9f} "
          f"resid_last={ceil:.9f}")
    check("all_mlp is NOT at the ceiling (it is the informative contrast)",
          abs(A["all_mlp"]["recovery_distractor"] - ceil) > 1e-6,
          f"all_mlp={100*A['all_mlp']['recovery_distractor']:+.1f}%")
    # the depth bands must carry real information: patching a SUBSET of mixers
    # leaves the other layers at host values, so a band cannot reach the ceiling.
    band_rec = [A[f"mix_band{b}"]["recovery_distractor"] for b in range(M.N_BANDS)]
    check("no single mixer band reaches the ceiling (bands are informative)",
          all(abs(r - ceil) > 1e-6 for r in band_rec),
          f"band recoveries {[f'{100*r:+.1f}%' for r in band_rec]}")
    check("the band arms differ from each other (depth matters)",
          max(band_rec) - min(band_rec) > 1e-6,
          f"spread {100*(max(band_rec)-min(band_rec)):.1f} pts")
    check("random-state donor gives a DIFFERENT result than the true donor",
          abs(A["rand_all_mix"]["delta_distractor"]
              - A["all_mix"]["delta_distractor"]) > 1e-9,
          f"rand={A['rand_all_mix']['delta_distractor']:+.6f} vs "
          f"hi={A['all_mix']['delta_distractor']:+.6f}")
    check("random donor is recorded with its provenance",
          A["rand_all_mix"]["rand_donor_state"] == s_b["state_id"]
          and A["rand_all_mix"]["rand_donor_game"] == "gameB")
    check("random donor game-mismatch flag is False for a different game",
          A["rand_all_mix"]["rand_donor_game_mismatch"] is False)
    check("partial overlap is counted in ROWS when action lengths differ",
          A["rand_all_mix"]["rand_n_patched"] == min(len(s_a["distractor_ids"]),
                                                     len(s_b["distractor_ids"]))
          and A["rand_all_mix"]["rand_n_host"] == len(s_a["distractor_ids"]),
          f"rows_patched={A['rand_all_mix']['rand_n_patched']} "
          f"host={A['rand_all_mix']['rand_n_host']}")
    check("n_patched counts row-writes summed over sites, not rows",
          A["rand_all_mix"]["n_patched_distractor"]
          == A["rand_all_mix"]["rand_n_patched"] * A["rand_all_mix"]["n_sites"],
          f"{A['rand_all_mix']['n_patched_distractor']} = "
          f"{A['rand_all_mix']['rand_n_patched']} rows x "
          f"{A['rand_all_mix']['n_sites']} sites")
    check("every site in the arm was actually written",
          A["all_mix"]["n_sites_patched_distractor"] == A["all_mix"]["n_sites"],
          f"{A['all_mix']['n_sites_patched_distractor']}/{A['all_mix']['n_sites']}")
    check("the correct action is scored by every arm too",
          all("delta_correct" in a for a in A.values() if not a.get("skipped")))
    check("Qwen-only split arms partition the mixer effect",
          "all_mix_full" in A and "all_mix_lin" in A)
    check("band arms cover every layer exactly once",
          sorted(l for b in range(M.N_BANDS) for l in A[f"mix_band{b}"]["layers"])
          == list(range(L)))

    # ── 4. hooks must not leak into a baseline pass ───────────────────────
    print("\n4. no leakage")
    patcher.begin_off(rows_lo)
    ids = lo["forced_distractor_ids"]
    with torch.no_grad():
        logits = adapter.forward(adapter.inputs(lo, "distractor", ids))
    lp_off, _ = H.action_logp(logits, rows_lo, tgt_lo)
    patcher.end()
    check("a pass with the patcher inactive reproduces the baseline exactly",
          abs(lp_off - rec["logp_lo"]["distractor"]) < 1e-9,
          f"{lp_off:.9f} vs {rec['logp_lo']['distractor']:.9f}")
    check("no hook fired while inactive", patcher.n_seen == 0
          and patcher.mode == "off")

    # ── 5. the real analyzer on the real record ───────────────────────────
    print("\n5. analyzer end-to-end on the produced record")
    with tempfile.TemporaryDirectory() as td:
        mod_dir = os.path.join(td, "per_cell_all", "text")
        os.makedirs(mod_dir)
        with open(os.path.join(mod_dir, f"{s_a['state_id']}__text.json"), "w") as f:
            json.dump(rec, f)
        # a second cell from another game, so the clustered bootstrap has 2 clusters
        rec_b, _ = H.run_cell(adapter, patcher, s_b, "text", arms, None, verify=False)
        with open(os.path.join(mod_dir, f"{s_b['state_id']}__text.json"), "w") as f:
            json.dump(rec_b, f)
        check("a cell with no random donor records the skip instead of failing",
              rec_b["arms"]["rand_all_mix"].get("skipped") is not None)

        import analyze_component_patch as AN
        AN.DEFAULT_ROOTS["qwen"] = td
        loaded, bad = AN.load_cells(sorted(glob_paths(mod_dir)))
        kept, gates = AN.gate_report(loaded, bad)
        check("both synthetic cells pass every gate", gates["n_kept"] == 2
              and gates["n_dropped"] == 0, json.dumps(gates["drop_reasons"]))
        block = AN.analyze_block("qwen", "all", "text",
                                 sorted(glob_paths(mod_dir)), n_boot=200)
        am = block["arms"]["all_mix"]
        check("analyzer recovers the same fraction as the raw record",
              abs(am["recovered_distractor_pct"]
                  - 100 * _ratio_of_means(kept, "all_mix")) < 1e-6,
              f"{am['recovered_distractor_pct']:.4f}%")
        check("analyzer reports a game-clustered CI with both games",
              am["n_games"] == 2 and am["n"] == 2, f"n={am['n']} games={am['n_games']}")
        check("ceiling arm is 100% in the analyzer too",
              abs(block["arms"]["resid_last"]["recovered_distractor_pct"] - 100.0) < 1e-4,
              f"{block['arms']['resid_last']['recovered_distractor_pct']:.6f}%")
        check("embed arm is 0% in the analyzer",
              block["arms"]["embed"]["mean_R_nats"] == 0.0)
        check("random arm is excluded from cross-model scope but still analysed",
              block["arms"]["rand_all_mix"]["n"] == 1,
              f"n={block['arms']['rand_all_mix']['n']} (one cell had no donor)")
        vs = AN.verdicts([block])[0]
        check("control verdicts are produced",
              vs["embed_is_zero"] and vs["self_patch_is_zero"] and vs["ceiling_is_one"],
              json.dumps({k: vs[k] for k in ("embed_is_zero", "self_patch_is_zero",
                                             "ceiling_is_one")}))
        csv_path = AN.write_csv([block], os.path.join(td, "summary.csv"))
        with open(csv_path) as f:
            lines = f.read().strip().split("\n")
        check("CSV has one row per arm plus a header",
              len(lines) == 1 + len(block["arms"]), f"{len(lines)} lines")

    # ── 6. resume-skip must keep the random-donor chain exact ─────────────
    print("\n6. resume-skip keeps the donor chain exact")
    s_c = make_state(nprng, "toyC__step1", "gameC")
    for s in (s_a, s_b, s_c):
        adapter.states[s["state_id"]] = s
    with tempfile.TemporaryDirectory() as td2:
        order3 = [s_a["state_id"], s_b["state_id"], s_c["state_id"]]
        rec_b2, _ = H.run_cell(adapter, patcher, s_b, "text", arms, None,
                               verify=False)
        with open(os.path.join(td2, f"{s_b['state_id']}__text.json"), "w") as f:
            json.dump(rec_b2, f)
        stats = H.run_shard(adapter, patcher, arms, [s_a, s_b, s_c], td2, "text",
                            order3, progress_every=99)
        check("run_shard counts one skip and two new cells",
              stats["skip"] == 1 and stats["ok"] == 2, json.dumps(stats))
        rc = json.load(open(os.path.join(td2, f"{s_c['state_id']}__text.json")))
        ra = rc["arms"]["rand_all_mix"]
        check("the cell after a skip gets the SKIPPED state's donor, not the "
              "last processed one",
              ra.get("rand_donor_state") == s_b["state_id"],
              f"donor={ra.get('rand_donor_state')}")
        check("and that donor is game-disjoint",
              ra.get("rand_donor_game_mismatch") is False)
        ra0 = json.load(open(os.path.join(td2, f"{s_a['state_id']}__text.json")))
        check("first state of a run still has no random donor",
              ra0["arms"]["rand_all_mix"].get("skipped") is not None)

    patcher.remove()
    print("\n" + "=" * 78)
    if FAILS:
        print(f"HARNESS SELF-TEST FAILED — {len(FAILS)} checks: {FAILS}")
        print("=" * 78)
        raise SystemExit(1)
    print("HARNESS SELF-TEST PASSED — gates are exact, alignment is index-based,")
    print("donors are distinguishable, and the analyzer reproduces the raw record")
    print("=" * 78)


def glob_paths(d):
    import glob
    return glob.glob(os.path.join(d, "*.json"))


def _ratio_of_means(kept, arm):
    """Independent recomputation of the metric, deliberately not sharing code
    with patch_spec, so a bug in the shared helper cannot hide here."""
    num = den = 0.0
    for _p, r in kept:
        a = r["arms"][arm]
        num += a["delta_distractor"]
        den += r["gap"]["distractor"]
    return num / den


if __name__ == "__main__":
    main()
