#!/usr/bin/env python3
"""
Layer-wise exposure slope (paper Fig. 4).

QUESTION
    At which layer does repeated exposure to a procedure start to become
    *action evidence*?  If you decode the action from the residual stream
    after layer l, how strongly does that decoding depend on k?

METHOD
    For every (state, k, modality) cell, teacher-force BOTH actions and read
    the hidden state after every one of the 32 language-model layers.  Apply
    the model's OWN final RMSNorm + LM head to the action positions only (a
    logit lens), then sum log-probs over the action tokens:

        s_l(a; k) = sum_{i in action tokens} log softmax(lens_l(h_{i-1}))[a_i]
        M_l(k)    = s_l(a_distractor; k) - s_l(a_correct; k)

    Per state and per layer we then fit  M_l(k) = alpha_l + beta_l * k  over
    k in {0,1,2,4,8} (analyze_layerwise_slope.py does the fit; this script only
    produces the raw per-layer log-probs).

    BOTH sides are stored separately, never only the margin: a fall in M can
    always be re-read as "the correct action collapsed" unless s_l(a_correct)
    is on disk.

WHY THIS IS CHEAP
    No attention weights are needed, so the forward runs on the memory-
    efficient sdpa path.  `output_hidden_states=True` costs 33 x [1,T,4096]
    bf16 ~= 4.4 GB at the visual length T~=16.5k, and the lens is applied to a
    slice of <=12 rows per layer, so the LM head is never run over the prompt.

SCORING CONVENTION
    logits at position i-1 predict token i; the scored span is
    [action_start, len(forced_ids)) with action_start from the
    lexical-interference stem-only tokenisation.  Character-offset counting
    (lib/position_utils.py::_stem_length) is off by one and drops the
    discriminative verb, under-measuring the slope by ~32%.  A smoke assertion
    compares our text `action_start` against
    `lib/text_positions.compute_text_positions`.

OUTPUT (resume-safe, one JSON per state, both modalities inside)
    results/layerwise-exposure-slope/per_layer/<state>.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import traceback
from typing import Dict, List, Optional, Sequence

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
MECH_ROOT = os.path.dirname(HERE)
if MECH_ROOT not in sys.path:
    sys.path.insert(0, MECH_ROOT)

from lib.config import (                                    # noqa: E402
    RESULTS_ROOT, LI_VARIANTS_DIR, MODEL_PATH, FORCED_THOUGHT, FORCED_STEM,
    K_LEVELS, FULL_ATTN_LAYERS,
)
from lib.cells import load_manifest, load_state              # noqa: E402
from lib.prompts_bridge import import_li_probe_lib_light     # noqa: E402
from lib.position_utils import _tokenize, _template_supports_thinking  # noqa: E402
from lib.visual_positions import build_visual_cell     # noqa: E402

OUT_DIR = os.path.join(RESULTS_ROOT, "layerwise-exposure-slope", "per_layer")
os.makedirs(OUT_DIR, exist_ok=True)

NUM_LAYERS = 32


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
def load():
    from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText
    from lib.model_guard import (assert_qwen35_backbone, image_pad_protocol,
                                 summarize)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = f"cuda:{os.environ.get('MECH_CUDA_INDEX', 0)}"
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    # sdpa: we need hidden states, not attention weights.
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="sdpa", device_map={"": device},
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # This experiment hard-codes NUM_LAYERS=32, FULL_ATTN_LAYERS and the
    # model.model.language_model.norm path.  Verify the checkpoint really is
    # Qwen3.5-9B before producing any artifact (see lib/model_guard.py).
    guard = assert_qwen35_backbone(model, model.config, strict=True)
    pad = image_pad_protocol(proc)
    line, guard_rec = summarize(guard, pad)
    print(f"[K] {line}", flush=True)
    if pad.get("problems"):
        raise RuntimeError(f"image-pad protocol check failed: {pad['problems']}")
    return model, proc, tok, device, guard_rec


def _final_norm_and_head(model):
    """Locate the model's own final RMSNorm and LM head (robust to wrapper
    naming across transformers versions)."""
    head = getattr(model, "lm_head", None)
    if head is None:
        raise RuntimeError("could not locate lm_head")
    for path in (("model", "language_model", "norm"),
                 ("model", "norm"),
                 ("language_model", "norm")):
        obj = model
        ok = True
        for attr in path:
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if ok:
            return obj, head
    raise RuntimeError("could not locate the final RMSNorm")


# fp32 copy of the LM head, built once per head object.  We slice <=12 rows per
# layer and want the lens read out exactly: at a verb logit of ~17 the bf16 ULP
# is 0.0625, which would quantise every per-token log-prob we report.  Hoisted
# out of the layer loop so the 151936x4096 upcast happens once, not 32x per
# forward.
_HEAD_W32: Dict[int, torch.Tensor] = {}


def _head_fp32(head):
    key = id(head)
    if key not in _HEAD_W32:
        w = head.weight.detach().float()
        b = head.bias.detach().float() if getattr(head, "bias", None) is not None else None
        _HEAD_W32[key] = (w, b)
    return _HEAD_W32[key]


# ---------------------------------------------------------------------------
# prompt rendering (mirrors lib/text_positions.py)
# ---------------------------------------------------------------------------
def _kwargs(tok):
    return ({"chat_template_kwargs": {"enable_thinking": False}}
            if _template_supports_thinking(tok) else {})


def _render_assistant(tok, messages, content: str) -> str:
    forced = list(messages) + [{"role": "assistant", "content": content}]
    return tok.apply_chat_template(forced, tokenize=False,
                                   continue_final_message=True,
                                   add_generation_prompt=False, **_kwargs(tok))


def build_text_cell(tok, probe_lib, state, k, body):
    messages = probe_lib.build_messages(state, "text", skill_text=body)
    stem_ids, _ = _tokenize(tok, _render_assistant(tok, messages, FORCED_STEM))
    action_start = len(stem_ids)
    ids_c, _ = _tokenize(tok, _render_assistant(
        tok, messages, f"{FORCED_THOUGHT}\nAction: {state['correct_action']}"))
    ids_d, _ = _tokenize(tok, _render_assistant(
        tok, messages, f"{FORCED_THOUGHT}\nAction: {state['distractor_action']}"))
    return {
        "messages": messages,
        "action_start": action_start,
        "stem_prefix_ok_c": ids_c[:action_start] == stem_ids,
        "stem_prefix_ok_d": ids_d[:action_start] == stem_ids,
        "forced_correct_ids": ids_c,
        "forced_distractor_ids": ids_d,
        "pixel_values": None,
        "image_grid_thw": None,
        "q_last_stem": action_start - 1,
    }


def build_visual_cell_wrapped(proc, tok, probe_lib, state, k):
    card_dir = os.path.join(LI_VARIANTS_DIR, "visual", state["state_id"], f"level_{k}")
    card_paths = sorted(glob.glob(os.path.join(card_dir, "card*.png")))
    if not card_paths:
        raise FileNotFoundError(f"no cards under {card_dir}")
    from PIL import Image
    images = [Image.open(p).convert("RGB") for p in card_paths]
    messages = probe_lib.build_messages(state, "visual", card_paths=card_paths)
    cell = build_visual_cell(processor=proc, tokenizer=tok, state=state, k=k,
                             messages=messages, images=images,
                             card_paths=card_paths)
    return {
        "messages": messages,
        "action_start": cell.action_start,
        "stem_prefix_ok_c": cell.stem_prefix_ok_c,
        "stem_prefix_ok_d": cell.stem_prefix_ok_d,
        "forced_correct_ids": cell.forced_correct_ids,
        "forced_distractor_ids": cell.forced_distractor_ids,
        "pixel_values": cell.pixel_values,
        "image_grid_thw": cell.image_grid_thw,
        "q_last_stem": cell.q_last_stem,
        "base_ids_match_processor": cell.base_ids_match_processor,
        "n_cards": len(card_paths),
        "debug": cell.debug,
    }


# ---------------------------------------------------------------------------
# the logit lens
# ---------------------------------------------------------------------------
@torch.inference_mode()
def per_layer_action_logp(model, norm, head, cell, forced_ids, device,
                          layers: Sequence[int], tok=None) -> Dict[int, List[float]]:
    """Sum of per-token log-probs of `forced_ids[action_start:]` decoded from
    the residual stream after each layer, using the model's own final norm +
    LM head.  Logits at position i-1 predict token i."""
    astart = cell["action_start"]
    T = len(forced_ids)
    if astart >= T:
        raise RuntimeError("action_start beyond the forced sequence")
    targets = forced_ids[astart:]
    # hidden rows that predict each target token
    rows = list(range(astart - 1, T - 1))
    if len(rows) != len(targets):
        raise RuntimeError(f"row/target mismatch {len(rows)} vs {len(targets)}")

    inputs = {
        "input_ids": torch.tensor([forced_ids], dtype=torch.long, device=device),
        "attention_mask": torch.ones(1, T, dtype=torch.long, device=device),
    }
    if cell["pixel_values"] is not None:
        inputs["pixel_values"] = cell["pixel_values"].to(device=device, dtype=torch.bfloat16)
        inputs["image_grid_thw"] = cell["image_grid_thw"].to(device=device)
        # Required alongside image_grid_thw since transformers 5.14, otherwise
        # M-RoPE cannot be computed and the forward raises.  We build model
        # kwargs by hand from sliced ids, so the processor's own field is not
        # available; lib.visual_positions derives the identical mask.
        import lib.visual_positions as VP
        inputs["mm_token_type_ids"] = VP.mm_token_type_ids_for(
            forced_ids, tok, device=device)

    out = model(output_hidden_states=True, use_cache=False, return_dict=True, **inputs)
    hs = out.hidden_states
    # hs[0] = embeddings, hs[l+1] = output of layer l
    if len(hs) != NUM_LAYERS + 1:
        raise RuntimeError(f"expected {NUM_LAYERS + 1} hidden states, got {len(hs)}")

    idx = torch.tensor(rows, dtype=torch.long, device=device)
    tgt = torch.tensor(targets, dtype=torch.long, device=device)
    W32, B32 = _head_fp32(head)
    result: Dict[int, List[float]] = {}
    for l in layers:
        h = hs[l + 1][0].index_select(0, idx).float()   # (n, D) — slice BEFORE norm
        h = norm(h)
        # head(h) would fail: h is fp32 while lm_head.weight is bf16.  Use the
        # hoisted fp32 head so the readout is exact rather than bf16-quantised.
        logits = torch.nn.functional.linear(h, W32, B32)   # (n, V)
        lp = torch.log_softmax(logits, dim=-1)
        result[int(l)] = lp.gather(1, tgt[:, None]).squeeze(1).tolist()
        del logits, lp, h
    del out, hs
    return result


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def run_state(sid, state, model, proc, tok, device, norm, head,
              modalities, ks, force=False, guard_rec=None):
    out_path = os.path.join(OUT_DIR, f"{sid}.json")
    if not force and os.path.exists(out_path):
        return "skip"

    probe_lib = import_li_probe_lib_light()
    layers = list(range(NUM_LAYERS))
    record: Dict[str, dict] = {}

    for mod in modalities:
        per_k = {}
        for k in ks:
            if mod == "text":
                vf = os.path.join(LI_VARIANTS_DIR, "text", sid, f"level_{k}.md")
                if not os.path.exists(vf):
                    continue
                with open(vf) as f:
                    body = f.read()
                cell = build_text_cell(tok, probe_lib, state, k, body)
            else:
                cell = build_visual_cell_wrapped(proc, tok, probe_lib, state, k)

            scores = {}
            for side, key in (("distractor", "forced_distractor_ids"),
                              ("correct", "forced_correct_ids")):
                fids = cell[key]
                pl = per_layer_action_logp(model, norm, head, cell, fids,
                                           device, layers, tok=tok)
                scores[side] = {
                    "per_layer_per_token": {str(l): pl[l] for l in layers},
                    "per_layer_sum": {str(l): float(sum(pl[l])) for l in layers},
                    "n_action_tokens": len(fids) - cell["action_start"],
                }
            per_k[str(k)] = {
                "action_start": cell["action_start"],
                "q_last_stem": cell["q_last_stem"],
                "stem_prefix_ok_c": cell["stem_prefix_ok_c"],
                "stem_prefix_ok_d": cell["stem_prefix_ok_d"],
                "n_action_tokens_distractor": scores["distractor"]["n_action_tokens"],
                "n_action_tokens_correct": scores["correct"]["n_action_tokens"],
                "scores": scores,
            }
        record[mod] = per_k

    payload = {
        "state_id": sid,
        "game_name": state.get("game_name"),
        "task_type": state.get("task_type"),
        "keyword": state.get("distractor_keyword"),
        "correct_action": state.get("correct_action"),
        "distractor_action": state.get("distractor_action"),
        "num_layers": NUM_LAYERS,
        "full_attn_layers": FULL_ATTN_LAYERS,
        "ks": list(ks),
        "lens": "model final RMSNorm + lm_head, applied to action rows only",
        "scoring": "logits[i-1] predicts token i; span [action_start, T)",
        "model_guard": guard_rec,
        "per_modality": record,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f)
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--states", nargs="*", default=None)
    ap.add_argument("--ks", nargs="*", type=int, default=K_LEVELS)
    ap.add_argument("--modalities", nargs="*", default=["text", "visual"],
                    choices=["text", "visual"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="2 states, assert action_start against text_positions")
    args = ap.parse_args()

    print("[K] loading model ...", flush=True)
    model, proc, tok, device, guard_rec = load()
    norm, head = _final_norm_and_head(model)
    print(f"[K] model on {device}; final norm = {type(norm).__name__}", flush=True)

    manifest = load_manifest()
    if args.states:
        manifest = [s for s in manifest if s["state_id"] in set(args.states)]
    if args.smoke:
        # Smoke runs text only, so its per-state JSON has no "visual" arm.  If
        # it landed in per_layer/ the full run would skip those states as
        # already-done and they would silently lose the visual modality, which
        # the paired text-minus-visual analysis needs.  Keep smoke artifacts in
        # their own directory.
        global OUT_DIR
        OUT_DIR = os.path.join(RESULTS_ROOT, "layerwise-exposure-slope", "per_layer_smoke")
        os.makedirs(OUT_DIR, exist_ok=True)
        manifest = manifest[:2]
    elif args.limit is not None:
        manifest = manifest[:args.limit]

    if args.smoke:
        _smoke_assert_action_start(tok, manifest, args.ks)

    n_ok = n_skip = n_err = 0
    t0 = time.time()
    for i, entry in enumerate(manifest):
        sid = entry["state_id"]
        try:
            state = load_state(sid)
            res = run_state(sid, state, model, proc, tok, device, norm, head,
                            args.modalities, args.ks, force=args.force,
                            guard_rec=guard_rec)
        except Exception as e:
            n_err += 1
            traceback.print_exc()
            with open(os.path.join(OUT_DIR, "errors.log"), "a") as f:
                f.write(f"{sid}\t{type(e).__name__}\t{e}\n")
            continue
        if res == "skip":
            n_skip += 1
        else:
            n_ok += 1
        if (i + 1) % 5 == 0:
            el = time.time() - t0
            print(f"[K] {i+1}/{len(manifest)} ({n_ok} ok / {n_skip} skip / "
                  f"{n_err} err) @ {el:.0f}s ({el/(i+1):.1f}s/state) "
                  f"mem={torch.cuda.max_memory_allocated()/2**30:.1f}GB",
                  flush=True)
    print(f"[K] done. ok={n_ok} skip={n_skip} err={n_err} @ {time.time()-t0:.0f}s")


def _smoke_assert_action_start(tok, manifest, ks):
    """GATE: our text action_start must equal lib.text_positions."""
    from lib.text_positions import compute_text_positions
    probe_lib = import_li_probe_lib_light()
    checked = 0
    for entry in manifest:
        sid = entry["state_id"]
        state = load_state(sid)
        for k in ks:
            vf = os.path.join(LI_VARIANTS_DIR, "text", sid, f"level_{k}.md")
            if not os.path.exists(vf):
                continue
            with open(vf) as f:
                body = f.read()
            ours = build_text_cell(tok, probe_lib, state, k, body)
            ref = compute_text_positions(tokenizer=tok, state=state, k=k,
                                          skill_body=body, variant_file=vf)
            assert ours["action_start"] == ref.debug["action_start"], (
                f"action_start mismatch {sid} k={k}: "
                f"{ours['action_start']} vs {ref.debug['action_start']}")
            assert ours["stem_prefix_ok_c"], f"stem_prefix_ok_c failed {sid} k={k}"
            assert ours["stem_prefix_ok_d"], f"stem_prefix_ok_d failed {sid} k={k}"
            assert ours["q_last_stem"] == ref.q_last_stem
            checked += 1
    print(f"[K] SMOKE GATE PASSED: action_start matches text_positions "
          f"on {checked} cells", flush=True)


if __name__ == "__main__":
    main()
