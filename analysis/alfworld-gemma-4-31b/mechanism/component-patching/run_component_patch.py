#!/usr/bin/env python3
"""
component-patching on Gemma-4-31B-it — architecture-agnostic component patching.

The Gemma half of a two-model experiment whose definition lives in README.md and
whose algorithm lives in patch_harness.py. This file implements ONLY the
Gemma-specific `ModelAdapter`. It is deliberately the mirror image of
`run_component_patch.py`: same arm list (imported from the same patch_spec.py, which
is a symlink to the Qwen tree's copy), same donor map, same gates, same metric.
What differs is architecture, and nothing else:

  * 60 layers read from the config at run time via `archlib_gemma.layer_plan`,
    never a hardcoded depth — 50 sliding + 10 full attention for this checkpoint.
  * EVERY layer's token mixer is `self_attn`; there is no second mixer class, so
    the `mix_L{L}` site is uniform. Qwen's site has to choose between
    `linear_attn` and `self_attn` per layer; that difference is exactly what the
    shared "Token Mixer" abstraction exists to paper over.
  * The visual path is the Gemma-4 processor protocol (`mm_token_type_ids`, no
    `image_grid_thw`, no manual pad expansion), reused from the validated
    `symbol-encoding/symlib_gemma.py`. The Qwen-VL pad arithmetic would be
    wrong here.
  * Sandwich norms: `post_attention_layernorm` / `post_feedforward_layernorm` sit
    between a module's output and the residual add. They give rise to two
    Gemma-only Appendix arms that bound the resulting magnitude attenuation
    (README §2.1). On Qwen those arms would be identical to the headline ones,
    which is why the headline uses the module output.
  * `final_logit_softcapping = 30.0` is applied by the model's own forward, so
    scores here are softcapped and Qwen's are not. Only the RATIO R/G is
    comparable across models, never the nats (README §6, §11).

MATERIAL — the same 165 lexical-interference decision states at k=0 and k=8
as the Qwen run, reached through `lib/config.py`, which points ANALYSIS_ROOT at
the Qwen tree. The documents are model-independent by construction and are never
re-generated here.

Run
---
  MECH_CUDA_INDEX=0 python3 run_component_patch.py --modality text --stage headline
  python3 run_component_patch.py --modality text --dry-run    # CPU, no weights
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
SYMBOL_ENCODING_DIR = os.path.join(MECH_ROOT, "symbol-encoding")
for p in (HERE, MECH_ROOT, SYMBOL_ENCODING_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from lib.config import (LI_VARIANTS_DIR, MODEL_PATH, RESULTS_ROOT)  # noqa: E402
from lib.cells import load_manifest, load_state                     # noqa: E402
from lib.prompts_bridge import import_li_probe_lib_light            # noqa: E402

import patch_spec as M                                                  # noqa: E402

ARCH = "gemma"
MODEL_NAME = "Gemma-4-31B-it"


def results_root(soft_tokens: int) -> str:
    """Resolution is an explicit arm: 280 matches the behavioural probes (the
    anchor this experiment must gate against), 1120 matches symbol-encoding. The two never share a directory, so they cannot be pooled by accident."""
    return os.path.join(RESULTS_ROOT, "component-patching", ARCH, f"soft{soft_tokens}")


# ══════════════════════════════════════════════════════════════════════════
# Adapter
# ══════════════════════════════════════════════════════════════════════════

class GemmaAdapter:
    arch = ARCH
    model_name = MODEL_NAME

    def __init__(self, device, soft_tokens=280, dry_run=False):
        self.device = device
        self.soft_tokens = int(soft_tokens)
        self.dry_run = dry_run
        self._probe_lib = import_li_probe_lib_light()
        self.guard_rec = None
        self.loc = None
        self.model = self.tok = self.processor = None

        import symlib as S
        import symlib_gemma
        symlib_gemma.patch(S, verbose=False)
        self.S = S
        self.SG = symlib_gemma
        import archlib_gemma as A
        self.A = A

        # Depth and mixer identity come from the config, never from constants:
        # Gemma-4 ships an explicit layer_types list and the last layer is forced
        # full. Reading it here (config only, no weights) lets --dry-run validate
        # the arm plan on CPU.
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
        self._cfg = cfg
        plan = A.layer_plan(cfg)
        self.plan = plan
        self.layer_types = list(plan["layer_types"])
        self.num_layers = int(plan["num_layers"])

    # -- model -------------------------------------------------------------
    def load_model(self, modality):
        import torch
        from transformers import (AutoModelForImageTextToText, AutoProcessor,
                                  AutoTokenizer)
        A = self.A
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required")
        self.tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(MODEL_PATH,
                                                       trust_remote_code=True)
        dmap, max_mem, note = _resolve_device_map(self.device)
        print(f"[M/gemma] placement: {note}", flush=True)
        kw = dict(torch_dtype=torch.bfloat16, trust_remote_code=True,
                  attn_implementation="sdpa", device_map=dmap)
        if max_mem is not None:
            kw["max_memory"] = max_mem
        self.model = AutoModelForImageTextToText.from_pretrained(MODEL_PATH, **kw)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        # No artifact is written before the architecture guard passes: a wrong
        # depth or a missing norm yields plausible-looking nonsense.
        self.guard_rec, plan, self.loc = A.guard(self.model, self.model.config)
        print(f"[M/gemma] {A.summarize(self.guard_rec)}", flush=True)
        if list(plan["layer_types"]) != self.layer_types:
            raise RuntimeError("config layer_types changed between load and guard")
        self.SG.assert_patched(self.S)
        # MoE or a per-layer-input branch would mean `layer.mlp` is NOT the whole
        # MLP contribution, silently invalidating the MLP arm. Assert it here
        # rather than in a caption.
        if getattr(self.model.config, "text_config", None) is not None:
            tcfg = self.model.config.text_config
            if getattr(tcfg, "enable_moe_block", False):
                raise RuntimeError("enable_moe_block is True: layer.mlp is not the "
                                   "whole MLP contribution; redefine the MLP site "
                                   "as mlp+experts before running")
            if getattr(tcfg, "hidden_size_per_layer_input", 0):
                raise RuntimeError("hidden_size_per_layer_input is set: the decoder "
                                   "layer has a third branch; site map is wrong")
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
        sid = state["state_id"]
        if modality == "text":
            messages = self._probe_lib.build_messages(state, "text",
                                                      skill_text=self._text_doc(sid, k))
            stem_ids = S.ids_of(self.tok, S.render_chat(self.tok, messages,
                                                        stem_only=True))
            astart = len(stem_ids)
            cell = {"action_start": astart, "messages": messages,
                    "n_image_tokens": None, "soft_tokens": None,
                    "card_paths": None, "images": None,
                    "batch_correct": None, "batch_distractor": None}
            for side, act in (("correct", state["correct_action"]),
                              ("distractor", state["distractor_action"])):
                ids = S.ids_of(self.tok, S.render_chat(self.tok, messages,
                                                       forced_action=act))
                cell[f"forced_{side}_ids"] = ids
                cell[f"stem_prefix_ok_{side[0]}"] = (ids[:astart] == stem_ids)
            return cell

        # Gemma-4 processor protocol: ask the processor for each render and use
        # its input_ids directly. No image_grid_thw, no merge_size, no manual pad
        # arithmetic — and therefore none of the Qwen-VL assumptions, which
        # fail on every Gemma visual cell.
        from PIL import Image
        paths = self._cards(sid, k)
        images = [Image.open(p).convert("RGB") for p in paths]
        messages = self._probe_lib.build_messages(state, "visual", card_paths=paths)
        self.SG.pin_soft_tokens(self.processor, self.soft_tokens)
        proc = self.processor
        stem_text = S.render_chat(proc, messages, stem_only=True)
        stem_batch = proc(text=[stem_text], images=images, return_tensors="pt")
        stem_ids = stem_batch["input_ids"][0].tolist()
        astart = len(stem_ids)
        mm = stem_batch.get("mm_token_type_ids")
        cell = {"action_start": astart, "messages": messages,
                "n_image_tokens": int((mm != 0).sum()) if mm is not None else None,
                "soft_tokens": self.soft_tokens, "card_paths": paths,
                "images": images, "batch_correct": None, "batch_distractor": None}
        for side, act in (("correct", state["correct_action"]),
                          ("distractor", state["distractor_action"])):
            text = S.render_chat(proc, messages, forced_action=act)
            batch = proc(text=[text], images=images, return_tensors="pt")
            ids = batch["input_ids"][0].tolist()
            cell[f"forced_{side}_ids"] = ids
            # Batches stay alive for the whole cell: every arm re-runs the full
            # forward and needs pixel_values again. layerwise-exposure-slope could drop them after
            # one scoring pass; component-patching cannot.
            cell[f"batch_{side}"] = batch
            cell[f"stem_prefix_ok_{side[0]}"] = (ids[:astart] == stem_ids)
        return cell

    def inputs(self, cell, side, ids):
        import torch
        batch = cell.get(f"batch_{side}")
        if batch is None:
            return {"input_ids": torch.tensor([list(ids)], dtype=torch.long,
                                              device=self.device),
                    "attention_mask": torch.ones(1, len(ids), dtype=torch.long,
                                                 device=self.device)}
        inp = self.SG._model_inputs_from_batch(batch, list(ids), self.device)
        return self.SG.align_mm_token_type_ids(inp, len(ids))

    def forward(self, inputs):
        import torch
        with torch.no_grad():
            try:
                out = self.model(use_cache=False, return_dict=True, **inputs)
            except TypeError:
                # the checkpoint may reject some processor key (symbol-encoding records
                # this per cell). Fall back to the minimal set.
                minimal = {k: inputs[k] for k in
                           ("input_ids", "attention_mask", "pixel_values",
                            "mm_token_type_ids")
                           if inputs.get(k) is not None}
                out = self.model(use_cache=False, return_dict=True, **minimal)
        return out.logits

    def reference_score(self, cell, state):
        """Gate 5: our own S_{k=0} against the probe-family scorer."""
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
        """No-op on purpose: the per-side processor batch is needed by every arm."""
        return None

    def release_all(self, cell):
        cell["batch_correct"] = cell["batch_distractor"] = None
        for im in (cell.get("images") or []):
            try:
                im.close()
            except Exception:
                pass
        cell["images"] = None

    # -- patch sites -------------------------------------------------------
    def sites(self, wanted):
        """name -> (module, takes_tuple).

        Gemma4TextAttention returns (hidden, attn_weights) — the decoder layer
        does `hidden_states, _ = self.self_attn(...)` — so takes_tuple is True for
        every mixer. Gemma4TextMLP, both post-norms, the decoder layer and the
        embedding all return a bare tensor. Patching the wrong element of a tuple
        would silently do nothing, which is why this is resolved once here.
        """
        A = self.A
        layers = self.loc["layers"]
        base = self.loc["layers_path"].rsplit(".", 1)[0]      # e.g. model.language_model
        embed = A._dig(self.model, tuple(base.split(".")) + ("embed_tokens",))
        if embed is None:
            raise RuntimeError(f"could not locate embed_tokens under {base}")
        all_sites = {M.EMBED_SITE: (embed, False)}
        for L, layer in enumerate(layers):
            if not hasattr(layer, "self_attn"):
                raise RuntimeError(f"layer {L} has no self_attn — not a Gemma4 text layer")
            if not hasattr(layer, "mlp"):
                raise RuntimeError(f"layer {L} has no mlp")
            all_sites[M.mix_site(L)] = (layer.self_attn, True)
            all_sites[M.mlp_site(L)] = (layer.mlp, False)
            all_sites[M.resid_site(L)] = (layer, False)
            # Gemma-only: the tensor actually added to the residual (README §2.1)
            if hasattr(layer, "post_attention_layernorm"):
                all_sites[M.postmix_site(L)] = (layer.post_attention_layernorm, False)
            if hasattr(layer, "post_feedforward_layernorm"):
                all_sites[M.postmlp_site(L)] = (layer.post_feedforward_layernorm, False)
        missing = [w for w in wanted if w not in all_sites]
        if missing:
            raise RuntimeError(f"arms reference unknown sites: {missing[:8]}")
        return {w: all_sites[w] for w in wanted}

    def provenance(self):
        p = self.plan
        return {
            "checkpoint": MODEL_PATH,
            "layers_path": (self.loc or {}).get("layers_path"),
            "norm_path": (self.loc or {}).get("norm_path"),
            "mixer_modules": "self_attn on every layer (50 sliding + 10 full)",
            "patch_point": "module output, before the residual add; on Gemma the "
                           "patched vector then passes post_attention_layernorm / "
                           "post_feedforward_layernorm (sandwich norm), which "
                           "Qwen3.5 does not have — see README §2.1 and the "
                           "*_postnorm Appendix arms",
            "forward": "single use_cache=False forward; no KV cache touched",
            "softcap": p.get("final_logit_softcapping"),
            "hidden_size": p.get("hidden_size"),
            "soft_tokens": self.soft_tokens,
            "layer_types": {"sliding_attention": p.get("n_sliding"),
                            "full_attention": p.get("n_full_attn")},
            "enable_moe_block": False,
            "hidden_size_per_layer_input": 0,
            "model_guard": self.guard_rec,
            "backend": _backend_note(),
        }


def _backend_note():
    try:
        import torch
        import transformers
        return {"transformers": transformers.__version__, "torch": torch.__version__,
                "python": sys.version.split()[0], "interpreter": sys.executable}
    except Exception:
        return {}


def _free_vram_gib(reserve_gib=1.5):
    import torch
    out = {}
    for i in range(torch.cuda.device_count()):
        free_b, _ = torch.cuda.mem_get_info(i)
        out[i] = max(0.0, free_b / 2 ** 30 - reserve_gib)
    return out


def _resolve_device_map(device):
    """MECH_DEVICE_MAP=single (default) puts the whole checkpoint on one card;
    =auto shards it across every visible GPU capped at what is ACTUALLY free, for
    when another job owns most of each card."""
    mode = os.environ.get("MECH_DEVICE_MAP", "single").lower()
    if mode in ("single", "", "none"):
        return {"": device}, None, f"single device {device}"
    free = _free_vram_gib(float(os.environ.get("MECH_VRAM_RESERVE_GIB", "1.5")))
    if not free:
        raise RuntimeError("MECH_DEVICE_MAP=auto but no CUDA devices visible")
    max_memory = {i: f"{g:.1f}GiB" for i, g in free.items() if g > 1.0}
    if not max_memory:
        raise RuntimeError(f"MECH_DEVICE_MAP=auto but nothing is free: {free}")
    note = ("sharded across " + ", ".join(f"cuda:{i}={v}" for i, v in max_memory.items())
            + f"  (total {sum(free.values()):.1f} GiB free)")
    return "auto", max_memory, note


# ══════════════════════════════════════════════════════════════════════════
# Driver — identical control flow to the Qwen runner
# ══════════════════════════════════════════════════════════════════════════

def contiguous_shard(order, shard, num_shards):
    if num_shards <= 1:
        return list(order)
    n = len(order)
    bounds = [round(i * n / num_shards) for i in range(num_shards + 1)]
    return list(order[bounds[shard]:bounds[shard + 1]])


def dry_run(modality, stage, limit, soft_tokens):
    print(f"[M/gemma] DRY RUN modality={modality} stage={stage} soft={soft_tokens}")
    adapter = GemmaAdapter("cpu", soft_tokens, dry_run=True)
    p = adapter.plan
    print(f"[M/gemma] L={adapter.num_layers} ({p['n_sliding']} sliding / "
          f"{p['n_full_attn']} full) softcap={p['final_logit_softcapping']} "
          f"H={p['hidden_size']}")
    arms = M.build_arms(adapter.num_layers, adapter.layer_types, ARCH, stage)
    wanted = M.required_sites(arms, adapter.num_layers, ARCH)
    print(f"[M/gemma] {len(arms)} arms, {len(wanted)} distinct sites")
    expect = ({M.EMBED_SITE}
              | {M.mix_site(l) for l in range(adapter.num_layers)}
              | {M.mlp_site(l) for l in range(adapter.num_layers)}
              | {M.resid_site(adapter.num_layers - 1)}
              | {M.postmix_site(l) for l in range(adapter.num_layers)}
              | {M.postmlp_site(l) for l in range(adapter.num_layers)})
    unknown = [w for w in wanted if w not in expect]
    print(f"[M/gemma] unknown site names: {unknown or 'none'}")
    if unknown:
        raise SystemExit(1)

    entries = load_manifest()
    order, dmap = M.ensure_donor_map(entries, ARCH)
    print(f"[M/gemma] donor map {M.donor_map_path()}\n"
          f"[M/gemma]   {dmap['n_states']} states / {dmap['n_games']} games "
          f"(SHARED with the Qwen run — same mismatched donors)")

    from transformers import AutoProcessor, AutoTokenizer
    adapter.tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if modality == "visual":
        adapter.processor = AutoProcessor.from_pretrained(MODEL_PATH,
                                                          trust_remote_code=True)
    import patch_harness as H
    n = limit or 3
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
        rows, targets = H.rows_and_targets(lo["forced_distractor_ids"],
                                           lo["action_start"])
        print(f"  {e['state_id'][:52]:52s} astart {lo['action_start']:5d}->"
              f"{hi['action_start']:5d} dT={hi['action_start']-lo['action_start']:+5d} "
              f"n_act={len(targets)} prefix={ok_pref} ids_match={ok_ids}"
              + (f" img={lo['n_image_tokens']}/{hi['n_image_tokens']}"
                 if modality == "visual" else ""))
        if not (ok_ids and ok_pref):
            bad += 1
        adapter.release_all(hi)
        adapter.release_all(lo)
    print(f"[M/gemma] DRY RUN {'FAILED' if bad else 'OK'} ({bad} bad cells of {n})")
    if bad:
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", choices=["text", "visual"], required=True)
    ap.add_argument("--stage", choices=["headline", "bands", "all"], default="headline")
    ap.add_argument("--soft-tokens", type=int, default=280,
                    help="visual resolution arm; 280 (default) matches the "
                         "behavioural probes, 1120 matches symbol-encoding")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--states", nargs="*", default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.dry_run:
        dry_run(args.modality, args.stage, args.limit, args.soft_tokens)
        return

    import patch_harness as H

    device = f"cuda:{os.environ.get('MECH_CUDA_INDEX', 0)}"
    entries = load_manifest()
    if args.states:
        keep = set(args.states)
        entries = [e for e in entries if e["state_id"] in keep]
    order, dmap = M.ensure_donor_map(load_manifest(), ARCH)
    shard_ids = set(contiguous_shard(order, args.shard, args.num_shards))
    entries = [e for e in entries if e["state_id"] in shard_ids]
    if args.limit is not None:
        # Slice the DONOR ORDER, never the manifest — see the identical comment in
        # the Qwen runner. A manifest-order prefix re-sorted by run_shard hands
        # some state a same-game random donor, which the analyzer's mismatch
        # gate then (correctly) refuses to pass.
        keep = set(order[:args.limit])
        entries = [e for e in entries if e["state_id"] in keep]

    print(f"[M/gemma] {MODEL_NAME} modality={args.modality} stage={args.stage} "
          f"soft={args.soft_tokens} k{M.K_HI}->k{M.K_LO} "
          f"shard {args.shard}/{args.num_shards}: {len(entries)} states "
          f"on {device}", flush=True)

    adapter = GemmaAdapter(device, args.soft_tokens)
    adapter.load_model(args.modality)
    arms = M.build_arms(adapter.num_layers, adapter.layer_types, ARCH, args.stage)
    wanted = M.required_sites(arms, adapter.num_layers, ARCH)
    patcher = H.RowPatcher(adapter.sites(wanted)).install()
    print(f"[M/gemma] {len(arms)} arms, {len(patcher.sites)} hooked sites "
          f"({dmap['n_games']} games in the shared donor map)", flush=True)

    out_dir = os.path.join(results_root(args.soft_tokens),
                           f"per_cell_{args.stage}", args.modality)
    t0 = time.time()
    try:
        stats = H.run_shard(adapter, patcher, arms, entries, out_dir,
                            args.modality, order, force=args.force,
                            verify_first=not args.no_verify)
    finally:
        patcher.remove()
    print(f"[M/gemma] FINISHED modality={args.modality} stage={args.stage} "
          f"soft={args.soft_tokens} {json.dumps(stats)} in {time.time()-t0:.0f}s "
          f"-> {out_dir}", flush=True)
    if stats["err"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
