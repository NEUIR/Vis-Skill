"""
component-patching shared patch harness — the algorithm both models run, in one file.

The two runners (`run_component_patch.py` for Qwen3.5-9B,
`run_component_patch.py` for Gemma-4-31B-it) implement ONLY a
`ModelAdapter`: how to build a cell (tokenizer / processor / prompt protocol),
where the patch sites are, and how to run one forward. Everything that defines
the experiment — which rows are patched, in what order, with which donor, which
gates must hold, what gets written — lives here, so the two models cannot drift.

WHY A SINGLE FULL FORWARD (no cache, no state surgery)
------------------------------------------------------
README §2 forbids touching GDN recurrent state, GDN conv state or the KV cache,
because those objects exist in only one of the two architectures. Patching
during one-token decode steps against a held cache would need
`snap_gdn` / `restore_gdn` / `crop_kv` — all Qwen-specific, and impossible to
port. Here every pass is one `use_cache=False` forward over the
whole sequence, and the hook replaces only the decision rows:

    rows = [action_start - 1, T - 1)      row i-1 predicts token i

Donor and host are aligned by ACTION-TOKEN INDEX, never by absolute position, so
the k=0 and k=8 prompts may differ in length freely (they do: different
documents, and on the visual arm a different number of image tokens).

Three consequences:
  * the `cache_consistency` caveat disappears — host, donor and patched passes
    are all the same kind of forward, so there is no chunk-vs-step discrepancy
    to tolerate;
  * `self_all` is exactly zero by construction, because the host's own
    activations are recorded in the same pass that produces its baseline;
  * nothing is architecture-specific, so Gemma's sliding-window cache and
    Qwen's GDN state are both simply never created.
"""
from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import torch

import patch_spec as M


# ══════════════════════════════════════════════════════════════════════════
# The patcher
# ══════════════════════════════════════════════════════════════════════════

class RowPatcher:
    """One forward hook per site; records or replaces the decision rows.

    `sites` maps site name -> (module, takes_tuple). `takes_tuple` records that
    the module returns (hidden, aux) rather than a bare tensor: patching element
    0 of the wrong shape would silently do nothing, so it is resolved once here.
    Qwen's full-attention layers and every Gemma attention layer return a tuple;
    Qwen's GatedDeltaNet, both models' MLPs, both models' decoder layers and the
    embedding return a bare tensor.

    Hooks stay installed for the whole run and no-op unless `active`, so a
    pass that forgot to configure the patcher can never corrupt a baseline.
    """

    def __init__(self, sites: Dict[str, Tuple[object, bool]]):
        self.sites = sites
        self.handles: List[object] = []
        self.mode = "off"            # 'record' | 'patch' | 'off'
        self.active = False
        self.rows: Optional[List[int]] = None
        self.store: Dict[str, torch.Tensor] = {}
        self.donor: Optional[Dict[str, torch.Tensor]] = None
        self.patch_sites: set = set()
        self.moved = 0.0
        self.n_patched = 0            # total row-writes summed over sites
        self.n_rows_patched = None    # rows written at each site (index-aligned)
        self.n_sites_patched = 0      # distinct sites actually written
        self.n_seen = 0

    # -- lifecycle ---------------------------------------------------------
    def install(self):
        for name, (mod, tup) in self.sites.items():
            self.handles.append(mod.register_forward_hook(self._make(name, tup)))
        return self

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def begin_record(self, rows: Sequence[int]):
        self.store = {}
        self.rows = list(rows)
        self.mode = "record"
        self.active = True
        self.n_seen = 0

    def begin_patch(self, rows: Sequence[int], sites: Sequence[str],
                    donor: Dict[str, torch.Tensor]):
        self.rows = list(rows)
        self.patch_sites = set(sites)
        self.donor = donor
        self.mode = "patch"
        self.active = True
        self.moved = 0.0
        self.n_patched = 0
        self.n_rows_patched = None
        self.n_sites_patched = 0
        self.n_seen = 0

    def begin_off(self, rows: Sequence[int]):
        self.rows = list(rows)
        self.mode = "off"
        self.active = False

    def end(self):
        self.active = False
        self.mode = "off"
        self.patch_sites = set()
        self.donor = None

    # -- the hook ----------------------------------------------------------
    def _make(self, name: str, takes_tuple: bool):
        def hook(_mod, _inp, out):
            if not self.active or self.rows is None:
                return out
            t = out[0] if takes_tuple else out
            if not torch.is_tensor(t) or t.dim() != 3:
                return out
            T = t.shape[1]
            rows = [r for r in self.rows if 0 <= r < T]
            if not rows:
                return out
            # device_map="auto" puts different layers on different GPUs, so the
            # row index has to follow the activation, not the caller's device.
            idx = torch.as_tensor(rows, dtype=torch.long, device=t.device)

            if self.mode == "record":
                self.store[name] = t.index_select(1, idx).detach().clone()
                self.n_seen += 1
                return out

            if self.mode == "patch" and name in self.patch_sites:
                d = self.donor.get(name) if self.donor else None
                if d is None:
                    return out
                n = min(d.shape[1], len(rows))
                if n <= 0:
                    return out
                sel = idx[:n]
                vals = d[:, :n, :].to(dtype=t.dtype, device=t.device)
                cur = t.index_select(1, sel)
                self.moved += float((cur.float() - vals.float()).abs().sum())
                self.n_patched += n
                self.n_sites_patched += 1
                # every site patches the same index-aligned row window, so a
                # disagreement means the donor store was built from a different
                # action span than the host's — refuse to average over it.
                if self.n_rows_patched is None:
                    self.n_rows_patched = n
                elif self.n_rows_patched != n:
                    raise RuntimeError(
                        f"site {name} patched {n} rows but another site patched "
                        f"{self.n_rows_patched}: donor/host action spans disagree")
                new = t.index_copy(1, sel, vals)
                return (new,) + tuple(out[1:]) if takes_tuple else new
            return out
        return hook


# ══════════════════════════════════════════════════════════════════════════
# Scoring
# ══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def action_logp(logits: torch.Tensor, rows: Sequence[int],
                targets: Sequence[int]) -> Tuple[float, List[float]]:
    """Sum log P over the action tokens, from the model's own logits.

    SLICE BEFORE THE FP32 CAST. Casting a whole [T, vocab] tensor costs
    T*vocab*4 bytes — ~15 GiB on a long Qwen image-pad prompt with a 152k vocab,
    and ~3 GiB per 1k tokens on Gemma's 262k vocab. Row i-1 predicts token i, so
    the rows we need are exactly `rows`; per-row log_softmax is unaffected by
    dropping the others, so this is numerically identical, just cheap.
    """
    idx = torch.as_tensor(list(rows), dtype=torch.long, device=logits.device)
    tgt = torch.as_tensor(list(targets), dtype=torch.long, device=logits.device)
    if idx.numel() != tgt.numel():
        raise ValueError(f"row/target mismatch {idx.numel()} vs {tgt.numel()}")
    sl = logits[0].index_select(0, idx).float()
    lp = torch.log_softmax(sl, dim=-1)
    per = lp.gather(1, tgt[:, None]).squeeze(1).tolist()
    del sl, lp
    return float(sum(per)), [float(x) for x in per]


def rows_and_targets(ids: Sequence[int], action_start: int) -> Tuple[List[int], List[int]]:
    """rows = the positions whose logits predict the action tokens."""
    T = len(ids)
    if action_start >= T:
        raise ValueError(f"action_start {action_start} beyond T {T}")
    rows = list(range(action_start - 1, T - 1))
    targets = list(ids[action_start:T])
    if len(rows) != len(targets):
        raise AssertionError("row/target misalignment")
    if action_start - 1 < 0:
        raise ValueError("action_start must be >= 1")
    return rows, targets


# ══════════════════════════════════════════════════════════════════════════
# One cell — the shared algorithm
# ══════════════════════════════════════════════════════════════════════════

def _donor_pass(adapter, patcher: RowPatcher, cell: dict, side: str) -> dict:
    """Forward one exposure, recording every site at the decision rows."""
    ids = cell[f"forced_{side}_ids"]
    rows, targets = rows_and_targets(ids, cell["action_start"])
    patcher.begin_record(rows)
    try:
        logits = adapter.forward(adapter.inputs(cell, side, ids))
        logp, per = action_logp(logits, rows, targets)
        del logits
    finally:
        patcher.end()
    return {"store": patcher.store, "logp": logp, "per_token": per,
            "rows": rows, "targets": targets, "n_rows": len(rows)}


def run_cell(adapter, patcher: RowPatcher, state: dict, modality: str,
             arms: Sequence[dict], rand_donor: Optional[dict],
             verify: bool = False) -> dict:
    """Full component-patching cell: donor k=8, host k=0, then every arm.

    `rand_donor` is the previous state's recorded donor store in this modality
    (a different game by construction), or None for the first state of a shard.
    """
    sid = state["state_id"]
    hi = adapter.build_cell(state, modality, M.K_HI)
    lo = adapter.build_cell(state, modality, M.K_LO)

    rec = {
        "model": adapter.model_name, "arch": adapter.arch,
        "state_id": sid, "game_name": state["game_name"],
        "task_type": state.get("task_type"),
        "keyword": state.get("distractor_keyword"),
        "modality": modality, "info_condition": M.INFO_CONDITION,
        "hi_k": M.K_HI, "lo_k": M.K_LO,
        "num_layers": adapter.num_layers, "layer_types": adapter.layer_types,
        "depth_rule": f"d = l/{adapter.num_layers - 1}",
        "action_start_hi": hi["action_start"], "action_start_lo": lo["action_start"],
        "delta_T": hi["action_start"] - lo["action_start"],
        "soft_tokens": hi.get("soft_tokens"),
        "n_image_tokens_hi": hi.get("n_image_tokens"),
        "n_image_tokens_lo": lo.get("n_image_tokens"),
        "checks": {}, "arms": {},
        "provenance": adapter.provenance(),
    }

    # ── per-cell assertions: the whole row-alignment argument rests on these ──
    ok_prefix = all(
        hi[f"stem_prefix_ok_{s[0]}"] and lo[f"stem_prefix_ok_{s[0]}"]
        for s in ("correct", "distractor"))
    ok_ids = all(hi[f"forced_{s}_ids"][hi["action_start"]:]
                 == lo[f"forced_{s}_ids"][lo["action_start"]:]
                 for s in ("correct", "distractor"))
    rec["checks"]["stem_prefix_ok"] = bool(ok_prefix)
    rec["checks"]["action_ids_match"] = bool(ok_ids)
    if not ok_prefix:
        raise RuntimeError(f"{sid}/{modality}: stem is not a token prefix "
                           f"(refusing to score a shifted span)")
    if not ok_ids:
        raise RuntimeError(f"{sid}/{modality}: k={M.K_HI} and k={M.K_LO} action "
                           f"token ids differ — index alignment is not legitimate")

    # ── donor (k=8) and host (k=0) passes ─────────────────────────────────
    donor, host, logp_hi, logp_lo, ntok = {}, {}, {}, {}, {}
    for side in ("distractor", "correct"):
        d = _donor_pass(adapter, patcher, hi, side)
        donor[side] = d["store"]
        logp_hi[side] = d["logp"]
        h = _donor_pass(adapter, patcher, lo, side)
        host[side] = h["store"]
        logp_lo[side] = h["logp"]
        ntok[side] = h["n_rows"]
        rec[f"rows_{side}"] = h["rows"]
        rec[f"per_token_hi_{side}"] = d["per_token"]
        rec[f"per_token_lo_{side}"] = h["per_token"]
        adapter.release(hi, side)

    rec["logp_hi"] = logp_hi
    rec["logp_lo"] = logp_lo
    rec["gap"] = {s: logp_hi[s] - logp_lo[s] for s in logp_hi}
    rec["margin_hi"] = logp_hi["distractor"] - logp_hi["correct"]
    rec["margin_lo"] = logp_lo["distractor"] - logp_lo["correct"]
    rec["n_action_tokens"] = ntok

    # ── reference consistency (gate 5): our own S_{k=0} vs the probe scorer ──
    if verify:
        ref = adapter.reference_score(lo, state)
        cc = {s: {"harness": logp_lo[s], "reference": ref[f"logp_{s}"],
                  "abs_diff": abs(logp_lo[s] - ref[f"logp_{s}"])} for s in logp_lo}
        cc["tol_nats"] = M.REFERENCE_TOL
        cc["ok"] = all(cc[s]["abs_diff"] <= M.REFERENCE_TOL for s in logp_lo)
        cc["note"] = ("both sides are single use_cache=False forwards over the same "
                      "tokens, so the only difference is hook overhead; there is "
                      "no chunk-vs-step term to tolerate")
        rec["checks"]["reference_consistency"] = cc

    # ── arm sweep ─────────────────────────────────────────────────────────
    g_d = rec["gap"]["distractor"]
    g_c = rec["gap"]["correct"]
    for arm in arms:
        name, sites, src = arm["name"], arm["sites"], arm["donor"]
        out = {"sites": arm["sites"], "donor": src, "kind": arm["kind"],
               "component": arm.get("component"), "scope": arm.get("scope", "both"),
               "n_sites": len(sites)}
        if arm.get("band") is not None:
            out["band"] = arm["band"]
            out["layers"] = arm["layers"]
            out["depth_range"] = arm["depth_range"]
        if src == "rand" and rand_donor is None:
            out["skipped"] = "no random donor for the first state of a shard"
            rec["arms"][name] = out
            continue
        for side in ("distractor", "correct"):
            ids = lo[f"forced_{side}_ids"]
            rows, targets = rows_and_targets(ids, lo["action_start"])
            if src == "hi":
                dstore = donor[side]
            elif src == "self":
                dstore = host[side]
            else:
                dstore = rand_donor[side]
            patcher.begin_patch(rows, sites, dstore)
            try:
                logits = adapter.forward(adapter.inputs(lo, side, ids))
                lp, _ = action_logp(logits, rows, targets)
                del logits
                out[f"moved_{side}"] = patcher.moved
                out[f"n_patched_{side}"] = patcher.n_patched
                out[f"n_rows_patched_{side}"] = patcher.n_rows_patched
                out[f"n_sites_patched_{side}"] = patcher.n_sites_patched
            finally:
                patcher.end()
            out[f"logp_{side}"] = lp
            out[f"delta_{side}"] = lp - logp_lo[side]
        out["margin"] = out["logp_distractor"] - out["logp_correct"]
        out["recovery_distractor"] = (out["delta_distractor"] / g_d) if abs(g_d) > 1e-9 else None
        out["recovery_correct"] = (out["delta_correct"] / g_c) if abs(g_c) > 1e-9 else None
        out["site_nonempty"] = out["moved_distractor"] > 0.0
        if src == "rand":
            out["rand_donor_state"] = rand_donor.get("_state_id")
            out["rand_donor_game"] = rand_donor.get("_game_name")
            out["rand_donor_game_mismatch"] = bool(
                rand_donor.get("_game_name") == state["game_name"])
            out["rand_n_patched"] = out["n_rows_patched_distractor"]
            out["rand_n_host"] = ntok["distractor"]
        rec["arms"][name] = out

    # ── gates that are properties of this cell ────────────────────────────
    ck = rec["checks"]
    em = rec["arms"]["embed"]
    ck["embed_null_ok"] = bool(abs(em["delta_distractor"]) < M.SELF_TOL
                               and em["moved_distractor"] == 0.0)
    sp = rec["arms"]["self_all"]
    ck["self_patch_delta"] = {"distractor": sp["delta_distractor"],
                              "correct": sp["delta_correct"]}
    ck["self_patch_ok"] = bool(abs(sp["delta_distractor"]) < M.SELF_TOL
                               and abs(sp["delta_correct"]) < M.SELF_TOL)
    rl = rec["arms"]["resid_last"]
    ck["ceiling_recovery"] = rl["recovery_distractor"]
    ck["ceiling_ok"] = bool(rl["recovery_distractor"] is not None
                            and abs(rl["recovery_distractor"] - 1.0) < M.CEILING_TOL)
    ck["all_sites_recorded"] = bool(
        len(donor["distractor"]) == len(patcher.sites)
        and len(host["distractor"]) == len(patcher.sites))

    # keep this cell's k=8 donor alive: it is the NEXT state's random control
    tagged = donor_keep(donor, state)
    adapter.release_all(hi)
    adapter.release_all(lo)
    donor.clear()
    host.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rec, tagged


def donor_keep(donor_store: Dict[str, Dict[str, torch.Tensor]], state: dict) -> dict:
    """Tag a donor store with provenance so the random-state control is auditable.

    Copies the per-side dicts but keeps the tensor references, so clearing the
    caller's dict does not free them. Holding exactly one of these between cells
    is what makes the random-state control free: state i's donor is state i-1's
    already-computed k=8 recording, no disk cache and no cross-process handshake.
    """
    out = {side: dict(st) for side, st in donor_store.items()}
    out["_state_id"] = state["state_id"]
    out["_game_name"] = state["game_name"]
    return out


# ══════════════════════════════════════════════════════════════════════════
# Driver loop shared by both runners
# ══════════════════════════════════════════════════════════════════════════

def _record_donor_only(adapter, patcher: RowPatcher, state: dict,
                       modality: str) -> dict:
    """Record this state's k=8 donor WITHOUT running any arm (2 forwards).

    Used on resume-skip. The random-state control of the NEXT state in the order
    is THIS state's recording; if a skipped cell simply `continue`d, the chain
    would hand the next processed state the donor of the last *processed* state
    instead of its order neighbour, silently breaking game-disjointness on a
    resumed run. Recording the donor keeps the chain exact for the ~10% of a
    cell's cost.
    """
    cell = adapter.build_cell(state, modality, M.K_HI)
    store = {}
    for side in ("distractor", "correct"):
        d = _donor_pass(adapter, patcher, cell, side)
        store[side] = d["store"]
        adapter.release(cell, side)
    adapter.release_all(cell)
    return donor_keep(store, state)


def run_shard(adapter, patcher: RowPatcher, arms: Sequence[dict],
              entries: Sequence[dict], out_dir: str, modality: str,
              order: Sequence[str], force: bool = False, verify_first: bool = True,
              progress_every: int = 2) -> dict:
    """Process one shard in `order`, carrying each cell's k=8 donor forward as
    the next cell's random-state control. Returns {ok, skip, err, seconds}.

    Sharding must be CONTIGUOUS over `order` (the driver does this) so that the
    previous state — the random donor — is in this shard's memory.
    """
    import json
    import traceback

    pos = {s: i for i, s in enumerate(order)}
    todo = sorted((e["state_id"] for e in entries), key=lambda s: pos.get(s, 1 << 30))

    os.makedirs(out_dir, exist_ok=True)
    prev_donor = None
    n_ok = n_skip = n_err = 0
    t0 = time.time()
    for i, sid in enumerate(todo):
        path = os.path.join(out_dir, f"{sid}__{modality}.json")
        if os.path.exists(path) and not force:
            n_skip += 1
            # keep the random-donor chain exact across the skip (see
            # _record_donor_only): the next state's control must be THIS state's
            # k=8 recording, which is what the shared donor map promises.
            state = adapter.load_state(sid)
            prev_donor = _record_donor_only(adapter, patcher, state, modality)
            continue
        state = adapter.load_state(sid)
        verify = verify_first and n_ok == 0
        try:
            rec, prev_donor = run_cell(adapter, patcher, state, modality, arms,
                                       prev_donor, verify=verify)
        except Exception:
            n_err += 1
            prev_donor = None      # a failed cell must not become anyone's donor
            with open(os.path.join(out_dir, "errors.log"), "a") as f:
                f.write(f"=== {sid} {modality}\n{traceback.format_exc()}\n")
            continue
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(rec, f)
        os.replace(tmp, path)
        n_ok += 1
        if verify:
            print(f"  [checks] {json.dumps(rec['checks'])[:900]}", flush=True)
        if i % progress_every == 0 or i == len(todo) - 1:
            el = time.time() - t0
            line = (f"  [{i+1}/{len(todo)}] ok={n_ok} skip={n_skip} err={n_err} "
                    f"{el:.0f}s ({el/max(1,n_ok):.1f}s/cell)")
            if torch.cuda.is_available():
                line += f" mem={torch.cuda.max_memory_allocated()/2**30:.1f}GB"
            print(line, flush=True)
    return {"ok": n_ok, "skip": n_skip, "err": n_err, "seconds": time.time() - t0}
