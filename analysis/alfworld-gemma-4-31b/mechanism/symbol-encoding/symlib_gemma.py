"""
Gemma-4-31B override for symlib's VISUAL scoring path.

WHY THIS FILE EXISTS
--------------------
`symlib.py` in this directory is identical to the Qwen tree's
`analysis/alfworld-qwen3.5-9b/mechanism/symbol-encoding/symlib.py`,
which produced the 9B results, so it is kept unchanged.

Its `score_actions_visual` is Qwen-VL specific and raises on Gemma-4:

    grid_thw  = base["image_grid_thw"]        -> KeyError: 'image_grid_thw'
    merge_len = processor.image_processor.merge_size ** 2   -> AttributeError

Without an override every Gemma visual cell (F2b `visual_casing` and the F4
visual readout) fails.  Both runners catch per-cell exceptions and exit 0, so
such a failure would only show up in `results/*/errors.log`; `assert_patched`
below turns it into a hard error.

Gemma-4's processor protocol is different in kind, not just in key names:

    Qwen-VL                          Gemma-4
    ----------------------------     ------------------------------------
    `image_grid_thw` (t,h,w)         absent
    `merge_size`                     absent (uses `max_soft_tokens`, `patch_size`)
    `<|vision_start|><|image_pad|>   `<|image|>` — ONE marker per image, and
     x N><|vision_end|>`, N computed   the processor expands it internally, so
     by hand via expand_image_pads     NO manual pad arithmetic is needed
    `pixel_values` (flat patches)    `pixel_values` (n_img, soft*4, 768),
                                     `mm_token_type_ids`, `image_position_ids`

So the Gemma path is actually SIMPLER: ask the processor for each rendered
variant and use its `input_ids` directly.

`max_soft_tokens` MATTERS
-------------------------
It controls how many tokens each image becomes, i.e. how legible the dense
skill cards are:

    280 (processor default)  -> ~254 tokens/image
    1120 (the top tier)          -> ~1075 tokens/image

The Gemma end-to-end runs that this experiment must be comparable to
(`../vis-skill`, SR 90.3) used **1120**, while the vLLM probe runs
used the **280** default because their
`start_vllm.sh` never passes `--mm-processor-kwargs`.  This module defaults to
1120 so the mechanism result sits on the same visual footing as the baseline,
and records the value actually used in every cell so the choice is auditable.
Override with `GEMMA_MAX_SOFT_TOKENS`.

WHAT IS *NOT* CHANGED
---------------------
Everything else is re-exported from the Qwen symlib untouched: `render_chat`,
`ids_of`, `tm_kwargs`, `score_actions_text`, `build_messages_visual`, the arm
constants and the analysis-side helpers.  The text arms (F1 `text_casing`,
F3b `case_matched`) already ran clean on Gemma — 1815 and 1980 cells, zero
errors — so the text path is left exactly as it is.

USAGE
-----
    import symlib as S
    import symlib_gemma
    symlib_gemma.patch(S)          # replaces S.score_actions_visual in place

`patch()` is idempotent and records what it replaced, so a caller can assert
the override is live before burning GPU hours.
"""
from __future__ import annotations

import os
from typing import List, Optional

import symlib as _S

# Re-export the model-independent helpers so callers can use this module as a
# drop-in for symlib if they prefer.
from symlib import (  # noqa: F401
    render_chat, ids_of, tm_kwargs, score_actions_text, build_messages_visual,
    FORCED_STEM, FORCED_THOUGHT,
)

MAX_SOFT_TOKENS = int(os.environ.get("GEMMA_MAX_SOFT_TOKENS", "1120"))


# ---------------------------------------------------------------------------
def _set_max_soft_tokens(processor, value: int) -> Optional[int]:
    """Pin the image processor's soft-token budget.  Returns the previous value
    (or None if the attribute does not exist)."""
    ip = getattr(processor, "image_processor", None)
    if ip is None or not hasattr(ip, "max_soft_tokens"):
        return None
    prev = int(ip.max_soft_tokens)
    ip.max_soft_tokens = int(value)
    return prev


def _model_inputs_from_batch(batch, ids: List[int], device, exclude=("input_ids",)):
    """Move a processor batch to the device, substituting our own token ids.

    Only tensor-valued keys are forwarded; anything the model does not accept is
    filtered by the caller's TypeError fallback (see score_actions_visual).
    """
    import torch
    out = {}
    for k, v in batch.items():
        if k in exclude:
            continue
        if hasattr(v, "to"):
            out[k] = v.to(device=device)
    out["input_ids"] = torch.tensor([ids], dtype=torch.long, device=device)
    out["attention_mask"] = torch.ones(1, len(ids), dtype=torch.long, device=device)
    return out


# ---------------------------------------------------------------------------
def score_actions_visual(model, tok, processor, messages, images,
                         correct: str, distractor: str, device,
                         max_soft_tokens: int = MAX_SOFT_TOKENS) -> dict:
    """Gemma-4 counterpart of symlib.score_actions_visual.

    Same contract, same returned keys, same scoring convention:
      * logits at row i-1 predict token i
      * the scored span is [action_start, len(ids))
      * action_start comes from the LI-style stem-only tokenisation, so the
        discriminative verb token is INSIDE the span
      * `stem_prefix_ok` asserts the stem is a literal prefix of both forced
        turns; a False here means the numbers are not comparable across arms

    Differences from the Qwen version:
      * no `image_grid_thw`, no `merge_size`, no manual pad expansion — the
        processor returns already-expanded `input_ids`
      * `max_soft_tokens` is pinned and reported
      * image-token count comes from `mm_token_type_ids`
    """
    import torch

    prev_mst = _set_max_soft_tokens(processor, max_soft_tokens)

    # ---- stem-only render defines action_start (the LI-style fix) ----------
    stem_text = render_chat(processor, messages, stem_only=True)
    stem_batch = processor(text=[stem_text], images=images, return_tensors="pt")
    stem_ids = stem_batch["input_ids"][0].tolist()
    action_start = len(stem_ids)

    mm = stem_batch.get("mm_token_type_ids")
    n_image_tokens = int((mm != 0).sum()) if mm is not None else None

    out = {
        "action_start": action_start,
        "stem_prefix_ok": True,
        "n_image_tokens": n_image_tokens,
        "max_soft_tokens": int(max_soft_tokens),
        "prev_max_soft_tokens": prev_mst,
        "processor_class": type(processor).__name__,
        "backend": "gemma4_symlib_gemma",
    }

    # ---- score both actions ------------------------------------------------
    for name, act in (("distractor", distractor), ("correct", correct)):
        forced_text = render_chat(processor, messages, forced_action=act)
        batch = processor(text=[forced_text], images=images, return_tensors="pt")
        ids = batch["input_ids"][0].tolist()

        if ids[:action_start] != stem_ids:
            out["stem_prefix_ok"] = False

        inputs = _model_inputs_from_batch(batch, ids, device)
        with torch.inference_mode():
            try:
                logits = model(use_cache=False, return_dict=True, **inputs).logits[0]
            except TypeError:
                # Older/newer checkpoints may not accept every processor key
                # (e.g. image_position_ids).  Retry with the minimal set rather
                # than failing the cell — but record that we had to.
                minimal = {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                    "pixel_values": inputs.get("pixel_values"),
                }
                minimal = {k: v for k, v in minimal.items() if v is not None}
                if "mm_token_type_ids" in inputs:
                    minimal["mm_token_type_ids"] = inputs["mm_token_type_ids"]
                logits = model(use_cache=False, return_dict=True, **minimal).logits[0]
                out[f"{name}_minimal_kwargs"] = True

            # Slice BEFORE the fp32 cast: a full [T, vocab] fp32 tensor is
            # enormous on image-heavy prompts (the Qwen version documents the
            # same ~15 GiB OOM trap).
            sl = logits[action_start - 1:len(ids) - 1].float()
            lp = torch.log_softmax(sl, dim=-1)
            per = [float(lp[i - action_start, ids[i]])
                   for i in range(action_start, len(ids))]
            del logits, sl, lp

        out[f"logp_{name}"] = float(sum(per))
        out[f"n_{name}"] = len(per)
        out[f"per_token_{name}"] = per
        out[f"T_{name}"] = len(ids)

    if prev_mst is not None:
        _set_max_soft_tokens(processor, prev_mst)
    return out


# ---------------------------------------------------------------------------
_PATCHED = []


def patch(module=None, verbose: bool = True) -> object:
    """Install the Gemma visual scorer onto `module` (default: the Qwen symlib
    imported as `symlib`).  Idempotent.  Returns the module."""
    m = module if module is not None else _S
    if getattr(m, "_gemma_visual_patched", False):
        return m
    original = getattr(m, "score_actions_visual", None)
    m.score_actions_visual = score_actions_visual
    m._gemma_visual_patched = True
    m._gemma_visual_original = original
    _PATCHED.append(getattr(m, "__name__", repr(m)))
    if verbose:
        print(f"[symlib_gemma] patched score_actions_visual on "
              f"{getattr(m, '__name__', m)} "
              f"(max_soft_tokens={MAX_SOFT_TOKENS}, "
              f"replaced={'yes' if original is not None else 'no'})",
              flush=True)
    return m


def assert_patched(module=None) -> None:
    """Hard gate: refuse to run if the override is not installed.

    Without this a Gemma run silently uses the Qwen implementation and fails
    every visual cell.
    """
    m = module if module is not None else _S
    if not getattr(m, "_gemma_visual_patched", False):
        raise RuntimeError(
            "symlib.score_actions_visual has NOT been patched for Gemma-4. "
            "Call symlib_gemma.patch(S) before scoring any visual cell, "
            "otherwise every cell dies with KeyError: 'image_grid_thw'.")


def pin_soft_tokens(processor, value: int = MAX_SOFT_TOKENS) -> int:
    """Pin the image processor's soft-token budget and return the value set.

    Gemma-4 turns each image into `max_soft_tokens`-scaled soft tokens; the
    processor default is 280 (~254 tokens/image), while the end-to-end Gemma
    baseline (`../vis-skill`, SR 90.3) ran at 1120 (~1075 tokens/image).  The
    mechanism experiments must match the baseline or the two are not measuring
    the same visual condition.
    """
    _set_max_soft_tokens(processor, value)
    return int(value)


def align_mm_token_type_ids(model_kwargs: dict, n_tokens: int):
    """Return model_kwargs with `mm_token_type_ids` resized to n_tokens.

    Gemma-4 requires `mm_token_type_ids` to be the same length as `input_ids`.
    Callers that append tokens to a prefilled prompt (e.g. F4's `score_words`,
    which re-runs `ids + word_ids` for multi-token candidates) would otherwise
    hit a shape mismatch.  Appended positions are text, so they pad with 0.
    Returns a NEW dict; the caller's dict is not mutated.
    """
    import torch
    out = dict(model_kwargs or {})
    mm = out.get("mm_token_type_ids")
    if mm is None:
        return out
    cur = mm.shape[-1]
    if cur == n_tokens:
        return out
    if cur > n_tokens:
        out["mm_token_type_ids"] = mm[..., :n_tokens].contiguous()
    else:
        pad = torch.zeros(*mm.shape[:-1], n_tokens - cur,
                          dtype=mm.dtype, device=mm.device)
        out["mm_token_type_ids"] = torch.cat([mm, pad], dim=-1)
    return out


def self_test(processor) -> dict:
    """CPU-only sanity check of the Gemma image protocol.

    Verifies the assumptions this module is built on, without needing a GPU:
    the processor accepts `<|image|>` markers, returns no `image_grid_thw`,
    exposes `mm_token_type_ids`, and honours `max_soft_tokens`.
    """
    from PIL import Image
    import numpy as np
    img = Image.fromarray((np.random.rand(400, 600, 3) * 255).astype("uint8"))
    txt = "<bos><|image|>\nhi"
    res = {}
    lo = processor(text=[txt], images=[img], return_tensors="pt")
    res["keys"] = sorted(lo.keys())
    res["has_image_grid_thw"] = "image_grid_thw" in lo
    res["has_mm_token_type_ids"] = "mm_token_type_ids" in lo
    res["len_at_default"] = int(lo["input_ids"].shape[1])

    prev = _set_max_soft_tokens(processor, 1120)
    hi = processor(text=[txt], images=[img], return_tensors="pt")
    res["len_at_1120"] = int(hi["input_ids"].shape[1])
    if prev is not None:
        _set_max_soft_tokens(processor, prev)
    res["max_soft_tokens_honoured"] = res["len_at_1120"] > res["len_at_default"]
    res["prev_max_soft_tokens"] = prev
    return res
