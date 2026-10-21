#!/usr/bin/env python3
"""
component-patching on Qwen3.5-9B — architecture-agnostic component patching.

The Qwen half of a two-model experiment whose definition lives in README.md and
whose algorithm lives in patch_harness.py. This file implements ONLY the
Qwen-specific `ModelAdapter`: how a cell is built (Qwen-VL image-pad expansion),
where the patch sites are (`linear_attn` for the 24 Gated DeltaNet layers,
`self_attn` for the 8 softmax layers, `mlp`, the decoder layer, the embedding),
and how one forward is run. The Gemma runner implements the same interface
against a different architecture; nothing about the intervention is defined here.

MATERIAL — the 165 LI decision states at k=0 and k=8, i.e. the same population
and the same documents layerwise-exposure-slope used, so Figure (a) and Figure (b) describe the
same states. component-patching is k=8 -> k=0 on the LI material, per README §3.

Run
---
  MECH_CUDA_INDEX=0 python3 run_component_patch.py --modality text   --stage headline
  MECH_CUDA_INDEX=1 python3 run_component_patch.py --modality visual --stage headline
  python3 run_component_patch.py --modality text --dry-run     # CPU, no weights
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MECH_ROOT = os.path.dirname(HERE)
ANALYSIS_ROOT = os.path.dirname(MECH_ROOT)
SYMBOL_ENCODING_DIR = os.path.join(MECH_ROOT, "symbol-encoding")
for p in (HERE, MECH_ROOT, SYMBOL_ENCODING_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from lib.config import LI_VARIANTS_DIR, LI_STATES_MANIFEST, MODEL_PATH  # noqa: E402
from lib.cells import load_manifest, load_state                         # noqa: E402
from lib.prompts_bridge import import_li_probe_lib_light                # noqa: E402

import patch_spec as M                                                      # noqa: E402

ARCH = "qwen"
MODEL_NAME = "Qwen3.5-9B"
RESULTS_ROOT = os.path.join(MECH_ROOT, "results", "component-patching", ARCH)


# ══════════════════════════════════════════════════════════════════════════
# Adapter
# ══════════════════════════════════════════════════════════════════════════

class QwenAdapter:
    """Qwen3.5-9B: 32 layers, 24 Gated DeltaNet + 8 full attention, no softcap."""

    arch = ARCH
    model_name = MODEL_NAME

    def __init__(self, device, dry_run=False):
        self.device = device
        self.dry_run = dry_run
        self._probe_lib = import_li_probe_lib_light()
        # layer_types come from config.json, NOT from hardcoded constants: the
        # whole point of component-patching is that depth and mixer identity are read from the
        # checkpoint, so the same code would be correct on a differently-shaped
        # Qwen. load_model() re-verifies against the live module tree.
        with open(_config_path()) as f:
            cfg = json.load(f)
        tc = cfg.get("text_config", cfg)
        self.layer_types = list(tc["layer_types"])
        self.num_layers = int(tc["num_hidden_layers"])
        self.model = self.tok = self.processor = None
        if not dry_run:
            import symlib as S
            self.S = S

    # -- model -------------------------------------------------------------
    def load_model(self, modality):
        S = self.S
        if modality == "text":
            self.model, self.tok = S.load_text_model(self.device)
        else:
            self.model, self.processor, self.tok = S.load_visual_model(self.device)
        cfg_lt = list(self.model.config.text_config.layer_types)
        if cfg_lt != self.layer_types:
            raise RuntimeError(
                f"config.json layer_types disagree with the loaded checkpoint: "
                f"{cfg_lt.count('full_attention')} vs "
                f"{self.layer_types.count('full_attention')} full-attention layers")
        return self.model

    # -- state / material --------------------------------------------------
    def load_state(self, sid):
        return load_state(sid)

    def _text_doc(self, sid, k):
        p = os.path.join(LI_VARIANTS_DIR, "text", sid, f"level_{k}.md")
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        with open(p, encoding="utf-8") as f:
            return f.read()

    def _cards(self, sid, k):
        d = os.path.join(LI_VARIANTS_DIR, "visual", sid, f"level_{k}")
        paths = sorted(glob.glob(os.path.join(d, "card*.png")))
        if not paths:
            raise FileNotFoundError(f"no cards under {d}")
        return paths

    def build_cell(self, state, modality, k):
        S = self.S
        tok = self.tok
        sid = state["state_id"]
        if modality == "text":
            messages = self._probe_lib.build_messages(state, "text",
                                                      skill_text=self._text_doc(sid, k))
            stem_ids = S.ids_of(tok, S.render_chat(tok, messages, stem_only=True))
            astart = len(stem_ids)
            cell = {"action_start": astart, "messages": messages, "extra": {},
                    "n_image_tokens": None, "soft_tokens": None,
                    "card_paths": None, "images": None}
            for side, act in (("correct", state["correct_action"]),
                              ("distractor", state["distractor_action"])):
                ids = S.ids_of(tok, S.render_chat(tok, messages, forced_action=act))
                cell[f"forced_{side}_ids"] = ids
                cell[f"stem_prefix_ok_{side[0]}"] = (ids[:astart] == stem_ids)
            return cell

        # visual: Qwen-VL emits ONE <|image_pad|> per image in the chat
        # template and the processor normally expands it. We tokenise text-only
        # so that action_start is exact in token space, then expand by hand with
        # the processor's own t*h*w/merge_len arithmetic (symlib.expand_image_pads).
        from PIL import Image
        paths = self._cards(sid, k)
        images = [Image.open(p).convert("RGB") for p in paths]
        messages = self._probe_lib.build_messages(state, "visual", card_paths=paths)
        proc = self.processor
        base = proc(text=[S.render_chat(proc, messages)], images=images,
                    return_tensors="pt")
        grid = base["image_grid_thw"]
        gl = grid.tolist()
        merge_len = int(proc.image_processor.merge_size) ** 2
        stem_ids = S.ids_of(tok, S.expand_image_pads(
            S.render_chat(proc, messages, stem_only=True), gl, merge_len))
        astart = len(stem_ids)
        import torch
        cell = {"action_start": astart, "messages": messages,
                "extra": {"pixel_values": base["pixel_values"].to(
                              device=self.device, dtype=torch.bfloat16),
                          "image_grid_thw": grid.to(self.device)},
                "n_image_tokens": int(sum(t * h * w // merge_len for t, h, w in gl)),
                "soft_tokens": None, "card_paths": paths, "images": images}
        for side, act in (("correct", state["correct_action"]),
                          ("distractor", state["distractor_action"])):
            ids = S.ids_of(tok, S.expand_image_pads(
                S.render_chat(proc, messages, forced_action=act), gl, merge_len))
            cell[f"forced_{side}_ids"] = ids
            cell[f"stem_prefix_ok_{side[0]}"] = (ids[:astart] == stem_ids)
        return cell

    def inputs(self, cell, side, ids):
        import torch
        n = len(ids)
        out = {"input_ids": torch.tensor([list(ids)], dtype=torch.long,
                                         device=self.device),
               "attention_mask": torch.ones(1, n, dtype=torch.long, device=self.device)}
        out.update(cell["extra"])
        return out

    def forward(self, inputs):
        import torch
        with torch.no_grad():
            out = self.model(use_cache=False, return_dict=True, **inputs)
        return out.logits

    def reference_score(self, cell, state):
        """The probe-family scorer, for gate 5. Single forward, no hooks."""
        S = self.S
        if cell["card_paths"] is None:
            return S.score_actions_text(self.model, self.tok, cell["messages"],
                                        state["correct_action"],
                                        state["distractor_action"], self.device)
        from PIL import Image
        imgs = [Image.open(p).convert("RGB") for p in cell["card_paths"]]
        try:
            return S.score_actions_visual(self.model, self.tok, self.processor,
                                          cell["messages"], imgs,
                                          state["correct_action"],
                                          state["distractor_action"], self.device)
        finally:
            for im in imgs:
                im.close()

    def release(self, cell, side):
        """Per-side cleanup hook. Qwen keeps one shared `extra` blob (pixel_values
        + image_grid_thw) for both forced turns, so there is nothing per-side to
        drop; the method exists because the Gemma adapter needs it and the
        harness must call the same interface on both models."""
        return None

    def release_all(self, cell):
        for im in (cell.get("images") or []):
            try:
                im.close()
            except Exception:
                pass
        cell["images"] = None
        cell["extra"] = {}

    # -- patch sites -------------------------------------------------------
    def sites(self, wanted):
        """name -> (module, takes_tuple). `wanted` restricts recording to the
        sites some arm actually uses."""
        import torch  # noqa: F401
        m = self.model.model
        for attr in ("language_model", "text_model"):
            if hasattr(m, attr):
                m = getattr(m, attr)
                break
        if not (hasattr(m, "layers") and hasattr(m, "embed_tokens")):
            raise RuntimeError(f"unexpected backbone {type(m)}")
        all_sites = {"embed": (m.embed_tokens, False)}
        for L, layer in enumerate(m.layers):
            # transformers names the flag layer_type (5.2) or block_type (5.14),
            # so the module that EXISTS decides, not the flag.
            is_full = hasattr(layer, "self_attn")
            mixer = layer.self_attn if is_full else layer.linear_attn
            all_sites[M.mix_site(L)] = (mixer, is_full)
            all_sites[M.mlp_site(L)] = (layer.mlp, False)
            all_sites[M.resid_site(L)] = (layer, False)
        # Qwen3.5 is pure pre-norm: the mixer/MLP output IS the tensor added to
        # the residual, so there is no post-norm site to define (README §2.1).
        missing = [w for w in wanted if w not in all_sites]
        if missing:
            raise RuntimeError(f"arms reference unknown sites: {missing[:8]}")
        return {w: all_sites[w] for w in wanted}

    def provenance(self):
        return {
            "checkpoint": MODEL_PATH,
            "layers_path": "model.language_model.layers",
            "mixer_modules": "linear_attn (GatedDeltaNet) or self_attn, chosen by "
                             "which attribute the layer has",
            "patch_point": "module output, before the residual add; Qwen3.5 is "
                           "pre-norm so this tensor is added raw",
            "forward": "single use_cache=False forward; no KV cache, no GDN state",
            "softcap": None,
            "layer_types": {"linear_attention": self.layer_types.count("linear_attention"),
                            "full_attention": self.layer_types.count("full_attention")},
            "backend": _backend_note(),
        }


def _config_path():
    return os.path.join(MODEL_PATH, "config.json")


def _backend_note():
    try:
        import torch
        import transformers
        return {"transformers": transformers.__version__, "torch": torch.__version__,
                "python": sys.version.split()[0],
                "interpreter": sys.executable}
    except Exception:
        return {}


# ══════════════════════════════════════════════════════════════════════════
# Driver
# ══════════════════════════════════════════════════════════════════════════

def contiguous_shard(order, shard, num_shards):
    """CONTIGUOUS blocks of the donor order, not a stride: the random-state
    control is the previous state in this order, so a stride would hand every
    state to a different shard and leave the control undefined."""
    if num_shards <= 1:
        return list(order)
    n = len(order)
    bounds = [round(i * n / num_shards) for i in range(num_shards + 1)]
    return list(order[bounds[shard]:bounds[shard + 1]])


def dry_run(modality, stage, limit):
    """CPU-only: validates material, tokenisation, row alignment and arm sites
    without loading 18 GB of weights. Everything except the forward pass."""
    print(f"[M/qwen] DRY RUN modality={modality} stage={stage}")
    adapter = QwenAdapter("cpu", dry_run=True)
    print(f"[M/qwen] L={adapter.num_layers} "
          f"({adapter.layer_types.count('linear_attention')} GDN / "
          f"{adapter.layer_types.count('full_attention')} full)")
    arms = M.build_arms(adapter.num_layers, adapter.layer_types, ARCH, stage)
    wanted = M.required_sites(arms, adapter.num_layers, ARCH)
    print(f"[M/qwen] {len(arms)} arms, {len(wanted)} distinct sites")
    # site names must be constructible from the layer plan alone
    expect = {"embed"} | {M.mix_site(l) for l in range(adapter.num_layers)} \
        | {M.mlp_site(l) for l in range(adapter.num_layers)} \
        | {M.resid_site(adapter.num_layers - 1)}
    unknown = [w for w in wanted if w not in expect]
    print(f"[M/qwen] unknown site names: {unknown or 'none'}")
    if unknown:
        raise SystemExit(1)

    entries = load_manifest()
    order, dmap = M.ensure_donor_map(entries, ARCH)
    print(f"[M/qwen] donor map {M.donor_map_path()}\n"
          f"[M/qwen]   {dmap['n_states']} states / {dmap['n_games']} games, "
          f"donor = previous state in the shared order, game-disjoint by construction")

    # tokenise a few cells for real: needs the tokenizer only
    import symlib as S
    from transformers import AutoTokenizer, AutoProcessor
    adapter.S = S
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    adapter.tok = tok
    if modality == "visual":
        adapter.processor = AutoProcessor.from_pretrained(MODEL_PATH,
                                                          trust_remote_code=True)
    import torch
    adapter.device = "cpu"
    n = limit or 4
    bad = 0
    for e in entries[:n]:
        state = load_state(e["state_id"])
        hi = adapter.build_cell(state, modality, M.K_HI)
        lo = adapter.build_cell(state, modality, M.K_LO)
        ok_ids = all(hi[f"forced_{s}_ids"][hi["action_start"]:]
                     == lo[f"forced_{s}_ids"][lo["action_start"]:]
                     for s in ("correct", "distractor"))
        ok_pref = all(hi[f"stem_prefix_ok_{s[0]}"] and lo[f"stem_prefix_ok_{s[0]}"]
                      for s in ("correct", "distractor"))
        rows, targets = harness_rows(lo)
        print(f"  {e['state_id'][:52]:52s} astart {lo['action_start']:5d}->{hi['action_start']:5d} "
              f"dT={hi['action_start']-lo['action_start']:+5d} "
              f"n_act={len(targets)} prefix={ok_pref} ids_match={ok_ids}"
              + (f" img={lo['n_image_tokens']}/{hi['n_image_tokens']}"
                 if modality == "visual" else ""))
        if not (ok_ids and ok_pref):
            bad += 1
        adapter.release_all(hi)
        adapter.release_all(lo)
    print(f"[M/qwen] DRY RUN {'FAILED' if bad else 'OK'} ({bad} bad cells of {n})")
    if bad:
        raise SystemExit(1)


def harness_rows(cell):
    import patch_harness as H
    return H.rows_and_targets(cell["forced_distractor_ids"], cell["action_start"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", choices=["text", "visual"], required=True)
    ap.add_argument("--stage", choices=["headline", "bands", "all"], default="headline",
                    help="headline = Figure b arms; bands = Appendix depth sweep; "
                         "all = both plus the Qwen-only full/GDN split")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--states", nargs="*", default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.dry_run:
        dry_run(args.modality, args.stage, args.limit)
        return

    import torch
    import patch_harness as H

    device = f"cuda:{os.environ.get('MECH_CUDA_INDEX', 0)}"
    entries = load_manifest()
    if args.states:
        keep = set(args.states)
        entries = [e for e in entries if e["state_id"] in keep]
    # the donor order is always built from the FULL manifest, never from the
    # shard or --states subset: both models and every shard must agree on it.
    order, dmap = M.ensure_donor_map(load_manifest(), ARCH)
    shard_ids = set(contiguous_shard(order, args.shard, args.num_shards))
    entries = [e for e in entries if e["state_id"] in shard_ids]
    if args.limit is not None:
        # Slice the DONOR ORDER, never the manifest. --limit N must mean "the
        # contiguous prefix of the shared order": that is the only subset for
        # which (a) both models see the same states and (b) the random-state
        # control's donor — the previously PROCESSED state — is still the
        # game-disjoint neighbour the map promises. Slicing manifest order breaks
        # (b): run_shard re-sorts by order position and hands some state a
        # same-game donor (the analyzer's mismatch gate checks this).
        keep = set(order[:args.limit])
        entries = [e for e in entries if e["state_id"] in keep]

    print(f"[M/qwen] {MODEL_NAME} modality={args.modality} stage={args.stage} "
          f"k{M.K_HI}->k{M.K_LO} shard {args.shard}/{args.num_shards}: "
          f"{len(entries)} states on {device}", flush=True)

    adapter = QwenAdapter(device)
    adapter.load_model(args.modality)
    arms = M.build_arms(adapter.num_layers, adapter.layer_types, ARCH, args.stage)
    wanted = M.required_sites(arms, adapter.num_layers, ARCH)
    patcher = H.RowPatcher(adapter.sites(wanted)).install()
    print(f"[M/qwen] {len(arms)} arms, {len(patcher.sites)} hooked sites; "
          f"random donor = previous state in the shared order "
          f"({dmap['n_games']} games)", flush=True)

    out_dir = os.path.join(RESULTS_ROOT, f"per_cell_{args.stage}", args.modality)
    t0 = time.time()
    try:
        stats = H.run_shard(adapter, patcher, arms, entries, out_dir,
                            args.modality, order, force=args.force,
                            verify_first=not args.no_verify)
    finally:
        patcher.remove()
    print(f"[M/qwen] FINISHED modality={args.modality} stage={args.stage} "
          f"{json.dumps(stats)} in {time.time()-t0:.0f}s -> {out_dir}", flush=True)
    if stats["err"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
