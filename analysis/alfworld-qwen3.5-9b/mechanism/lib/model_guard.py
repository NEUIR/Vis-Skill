"""
Model-identity guard for the mechanism experiments.

WHY THIS EXISTS
---------------
The mechanism code is written against **Qwen3.5-9B** and hard-codes facts about
it: 32 LM layers, full softmax attention only at layers [3,7,11,15,19,23,27,31]
(the other 24 are Gated DeltaNet / linear attention), 16 query heads with GQA
over 4 KV heads, head_dim 256, hidden 4096, and the Qwen-VL image-pad protocol.

Experiments that need `AutoModelForImageTextToText` plus a processor for the
multimodal cells build the model themselves. Without a check, pointing
`MECH_MODEL` at a different checkpoint would produce silently wrong numbers
rather than an error.

This module validates `FULL_ATTN_LAYERS` against the model's own
`config.text_config.layer_types` for any loader, plus checks for the
multimodal wrapper path and the image-pad protocol that
`lib/visual_positions.py` depends on.

Call `assert_qwen35_backbone(model, config)` right after `from_pretrained`.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from .config import FULL_ATTN_LAYERS, NUM_HEADS, HEAD_DIM, HIDDEN_SIZE

EXPECTED_NUM_LAYERS = 32
EXPECTED_LINEAR_LAYERS = EXPECTED_NUM_LAYERS - len(FULL_ATTN_LAYERS)


def text_config(config):
    """Qwen3.5 multimodal checkpoints nest the LM config under `text_config`."""
    return getattr(config, "text_config", config)


def read_layer_types(config) -> Optional[List[str]]:
    tc = text_config(config)
    lt = getattr(tc, "layer_types", None)
    if lt is None:
        return None
    return list(lt)


def assert_qwen35_backbone(model, config, *, strict: bool = True) -> dict:
    """Verify the loaded checkpoint matches the architecture this code assumes.

    Returns a dict of what was checked (useful to log into result records).
    Raises RuntimeError on any mismatch when `strict=True`.

    Checks, in order of how badly they would corrupt results:
      1. layer count == 32
      2. `layer_types` marks exactly FULL_ATTN_LAYERS as full_attention
         (this is what makes the DeltaNet/full-attn band shading in the figures
         and the "only 8 layers are ablatable" caveat true)
      3. head geometry: num_attention_heads, head_dim, hidden_size
      4. GQA: num_key_value_heads divides num_attention_heads (head -> kv head
         is mapped by integer division)
      5. the multimodal wrapper path `model.model.language_model.layers` exists
    """
    problems: List[str] = []
    tc = text_config(config)
    found: dict = {}

    # 1. layer count
    n_layers = getattr(tc, "num_hidden_layers", None)
    found["num_hidden_layers"] = n_layers
    if n_layers is not None and int(n_layers) != EXPECTED_NUM_LAYERS:
        problems.append(
            f"num_hidden_layers={n_layers}, expected {EXPECTED_NUM_LAYERS}")

    # 2. layer types
    lt = read_layer_types(config)
    if lt is None:
        # Not fatal: some checkpoints omit layer_types.  Warn only.
        found["layer_types"] = None
        if strict:
            problems.append(
                "config has no `layer_types`; cannot confirm which layers are "
                "full attention. The DeltaNet/full-attn split this code assumes "
                "is unverified.")
    else:
        found["layer_types_n"] = len(lt)
        full = [i for i, t in enumerate(lt) if t == "full_attention"]
        found["full_attn_layers_in_config"] = full
        if full != FULL_ATTN_LAYERS:
            problems.append(
                f"full-attention layers in config = {full}, but this code "
                f"assumes {FULL_ATTN_LAYERS}")
        n_lin = len(lt) - len(full)
        found["linear_attn_layers"] = n_lin
        if n_lin != EXPECTED_LINEAR_LAYERS:
            problems.append(
                f"{n_lin} non-full-attention layers, expected "
                f"{EXPECTED_LINEAR_LAYERS}")

    # 3. head geometry
    for attr, expected in (("num_attention_heads", NUM_HEADS),
                           ("head_dim", HEAD_DIM),
                           ("hidden_size", HIDDEN_SIZE)):
        got = getattr(tc, attr, None)
        found[attr] = got
        if got is not None and int(got) != expected:
            problems.append(f"{attr}={got}, expected {expected}")

    # 4. GQA divisibility (visual_ablate maps head -> kv head by h // n_rep)
    n_q = getattr(tc, "num_attention_heads", None)
    n_kv = getattr(tc, "num_key_value_heads", None)
    found["num_key_value_heads"] = n_kv
    if n_q and n_kv and int(n_q) % int(n_kv) != 0:
        problems.append(
            f"num_attention_heads={n_q} is not divisible by "
            f"num_key_value_heads={n_kv}; the GQA head->kv-head mapping in "
            f"the attention-patching code would be wrong")

    # 5. multimodal wrapper path
    lm = getattr(getattr(model, "model", None), "language_model", None)
    found["has_language_model_wrapper"] = lm is not None
    if lm is None:
        # A text-only checkpoint is acceptable for layerwise-exposure-slope's text arm, but
        # visual cells need the wrapper.  Warn rather than fail.
        found["wrapper_note"] = (
            "model.model.language_model missing — visual cells will fail; "
            "text-only use is fine")
    else:
        layers = getattr(lm, "layers", None)
        found["n_layer_modules"] = len(layers) if layers is not None else None
        if layers is not None and len(layers) != EXPECTED_NUM_LAYERS:
            problems.append(
                f"model.model.language_model.layers has {len(layers)} modules, "
                f"expected {EXPECTED_NUM_LAYERS}")
        if layers is not None:
            missing = [l for l in FULL_ATTN_LAYERS
                       if not hasattr(layers[l], "self_attn")]
            if missing:
                problems.append(
                    f"layers {missing} have no `self_attn` module; the sdpa "
                    f"patch in the attention-patching code cannot hook them")

    found["problems"] = problems
    if problems and strict:
        raise RuntimeError(
            "Loaded checkpoint does not match the Qwen3.5-9B architecture this "
            "code is written for:\n  - " + "\n  - ".join(problems) +
            "\nSet MECH_MODEL to the Qwen3.5-9B checkpoint, or pass "
            "strict=False to proceed anyway and record the mismatch."
        )
    return found


def image_pad_protocol(processor) -> dict:
    """Verify the processor exposes what lib/visual_positions.py needs."""
    found: dict = {}
    ip = getattr(processor, "image_processor", None)
    found["has_image_processor"] = ip is not None
    if ip is None:
        found["problems"] = ["processor has no image_processor"]
        return found
    merge = getattr(ip, "merge_size", None)
    patch = getattr(ip, "patch_size", None)
    found["merge_size"] = merge
    found["patch_size"] = patch
    if merge is None:
        found["problems"] = [
            "image_processor has no `merge_size`; expand_image_pads cannot "
            "compute the pad count per image"]
    else:
        found["problems"] = []
    tok = getattr(processor, "tokenizer", None)
    if tok is not None:
        try:
            found["image_pad_id"] = tok.convert_tokens_to_ids("<|image_pad|>")
        except Exception as e:                       # pragma: no cover
            found["image_pad_id_error"] = str(e)
    return found


def summarize(model_info: dict, pad_info: Optional[dict] = None) -> Tuple[str, dict]:
    """One-line human summary + a JSON-safe dict for result records."""
    n = model_info.get("num_hidden_layers")
    fa = model_info.get("full_attn_layers_in_config")
    line = (f"backbone: {n} layers, full-attn at {fa}, "
            f"{model_info.get('num_attention_heads')} heads / "
            f"{model_info.get('num_key_value_heads')} kv-heads, "
            f"head_dim {model_info.get('head_dim')}")
    rec = {"model_check": model_info}
    if pad_info is not None:
        rec["image_pad_check"] = pad_info
        line += f" | merge_size {pad_info.get('merge_size')}"
    return line, rec
