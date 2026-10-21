"""
VISUAL cell positions — the multimodal counterpart of `lib/text_positions.py`.

`lib/position_utils.py::_stem_length` counts stem tokens by *character offsets* in
the combined forced text. It is off by one: the token that straddles the
base/forced boundary gets counted into the stem, so `action_start` shifts one
token right and the first action token — the discriminative VERB — is dropped
from the scored span. The text path avoids this by mirroring
lexical-interference's `probe_lib.score_action`:

    action_start = len( tokenize( messages + [assistant: FORCED_STEM] ) )

Any visual experiment that SCORES actions must use the same convention, or its
baseline will not reproduce the lexical-interference visual anchor (0.552) and
nothing downstream is interpretable.

This module applies the stem-only convention in processor/visual space:

    1. render the stem-only assistant turn (`FORCED_STEM`, no action)
    2. expand the image-pad placeholders exactly like `Qwen3VLProcessor`
    3. action_start = len(tokenize(expanded stem-only text))
    4. assert `stem_prefix_ok`: the stem tokenisation is a literal prefix of
       each full forced tokenisation (correct and distractor)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .config import FORCED_THOUGHT, FORCED_STEM

IMAGE_TOKEN = "<|image_pad|>"
VISION_BLOCK = "<|vision_start|>" + IMAGE_TOKEN + "<|vision_end|>"


def mm_token_type_ids_for(input_ids, tokenizer, device=None):
    """Build the `mm_token_type_ids` that transformers >= 5.14 demands.

    Passing `image_grid_thw` without `mm_token_type_ids` now raises
        "Multimodal data was passed ... but `mm_token_type_ids` is missing.
         Please pass `mm_token_type_ids` ... so that multimodal RoPE (M-RoPE)
         can be computed correctly."
    The experiments here slice the processor's ids down to the prefill, so they
    build the model kwargs by hand and never carried this field. For
    Qwen3.5-VL the processor's own value is exactly
    `(input_ids == <|image_pad|>)` (identical to
    `AutoProcessor(...)(text=..., images=...)["mm_token_type_ids"]` on every
    image position), so deriving it from the sliced ids is equivalent, not an
    approximation.
    """
    import torch as _torch
    img_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
    ids = input_ids if _torch.is_tensor(input_ids) else _torch.tensor(input_ids)
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
    mm = (ids == img_id).long()
    return mm.to(device) if device is not None else mm


@dataclass
class VisualCell:
    """Everything needed to score / ablate one (state, k, visual) cell."""
    state_id: str
    k: int
    # --- token space ---
    input_ids_stem: List[int]          # prefill through q_last_stem (attention reads)
    forced_correct_ids: List[int]      # full forced sequence, correct action
    forced_distractor_ids: List[int]   # full forced sequence, distractor action
    action_start: int
    L_prefix: int
    stem_len: int
    q_last_stem: int
    q_first_disc: int
    action_tokens_correct: List[int]
    action_tokens_distractor: List[int]
    n_action_tokens: int
    # --- image space ---
    card_paths: List[str]
    image_spans: List[Tuple[int, int]]      # (start, n_image_tokens) per card, in prefill coords
    pixel_values: object                    # torch tensor from the processor
    image_grid_thw: object
    # --- checks ---
    stem_prefix_ok_c: bool
    stem_prefix_ok_d: bool
    base_ids_match_processor: bool
    debug: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# chat-template rendering
# ---------------------------------------------------------------------------
_TM_KWARGS_CACHE: Dict[int, dict] = {}


def _template_kwargs(tokenizer) -> dict:
    tid = id(tokenizer)
    if tid not in _TM_KWARGS_CACHE:
        tmpl = getattr(tokenizer, "chat_template", "") or ""
        _TM_KWARGS_CACHE[tid] = (
            {"chat_template_kwargs": {"enable_thinking": False}}
            if "enable_thinking" in tmpl else {}
        )
    return _TM_KWARGS_CACHE[tid]


def render_base(processor, messages) -> str:
    """Prompt with add_generation_prompt=True (no assistant turn)."""
    return processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        **_template_kwargs(processor.tokenizer),
    )


def render_assistant(processor, messages, content: str) -> str:
    """Prompt with a continued assistant turn carrying exactly `content`."""
    forced = list(messages) + [{"role": "assistant", "content": content}]
    return processor.apply_chat_template(
        forced, tokenize=False, continue_final_message=True,
        add_generation_prompt=False, **_template_kwargs(processor.tokenizer),
    )


def render_forced_action(processor, messages, action: str) -> str:
    return render_assistant(processor, messages, f"{FORCED_THOUGHT}\nAction: {action}")


# ---------------------------------------------------------------------------
# image-pad expansion (token-for-token identical to Qwen3VLProcessor.__call__)
# ---------------------------------------------------------------------------
def expand_image_pads(rendered: str, grid_thw: Sequence[Sequence[int]],
                      merge_len: int) -> str:
    """Expand each single-pad vision block to its true pad count.

    Must split on the WHOLE `<|vision_start|>…<|vision_end|>` block: a naive
    `str.replace(IMAGE_TOKEN, IMAGE_TOKEN*n, 1)` finds its next "first
    occurrence" INSIDE the already-expanded run and piles every image's pads
    into image 1.
    """
    parts = rendered.split(VISION_BLOCK)
    if len(parts) - 1 != len(grid_thw):
        raise RuntimeError(
            f"{len(parts) - 1} vision blocks in template vs {len(grid_thw)} grids"
        )
    out = parts[0]
    for i, thw in enumerate(grid_thw):
        t, h, w = (int(x) for x in thw)
        n = t * h * w // merge_len
        out += ("<|vision_start|>" + IMAGE_TOKEN * n + "<|vision_end|>" + parts[i + 1])
    return out


def _tok(tokenizer, text: str) -> List[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------
def build_visual_cell(
    *,
    processor,
    tokenizer,
    state: dict,
    k: int,
    messages: list,
    images: list,
    card_paths: List[str],
) -> VisualCell:
    """Build one corrected visual cell.

    `messages` must already be assembled by LI's `probe_lib.build_messages(
    state, "visual", card_paths=...)` so the prompt is byte-identical to the
    main runs. `images` are the opened PIL cards in the same order as
    `card_paths`.
    """
    # ---- base prompt through the processor (authoritative tokenisation) ----
    rendered_base = render_base(processor, messages)
    base_inputs = processor(text=[rendered_base], images=images, return_tensors="pt")
    grid_thw = base_inputs["image_grid_thw"]
    base_ids_proc = base_inputs["input_ids"][0].tolist()
    L_prefix = len(base_ids_proc)

    merge_len = int(processor.image_processor.merge_size) ** 2
    grids = grid_thw.tolist()

    exp_base = expand_image_pads(rendered_base, grids, merge_len)
    base_ids = _tok(tokenizer, exp_base)
    base_ids_match = (base_ids == base_ids_proc)

    # ---- THE FIX: stem-only assistant turn, tokenised as its own sequence ----
    exp_stem = expand_image_pads(render_assistant(processor, messages, FORCED_STEM),
                                 grids, merge_len)
    stem_ids = _tok(tokenizer, exp_stem)
    action_start = len(stem_ids)
    stem_len = action_start - L_prefix

    correct = state["correct_action"]
    distractor = state["distractor_action"]
    exp_fc = expand_image_pads(render_forced_action(processor, messages, correct),
                               grids, merge_len)
    exp_fd = expand_image_pads(render_forced_action(processor, messages, distractor),
                               grids, merge_len)
    ids_fc = _tok(tokenizer, exp_fc)
    ids_fd = _tok(tokenizer, exp_fd)

    # LI guard: the stem tokenisation must be a literal prefix of both turns.
    stem_prefix_ok_c = (ids_fc[:action_start] == stem_ids)
    stem_prefix_ok_d = (ids_fd[:action_start] == stem_ids)

    q_last_stem = action_start - 1
    at_c = ids_fc[action_start:]
    at_d = ids_fd[action_start:]
    n_action_tokens = max(len(at_c), len(at_d))

    q_first_disc = q_last_stem
    for i, (a, b) in enumerate(zip(at_c, at_d)):
        if a != b:
            q_first_disc = action_start + i - 1
            break

    prefill_end = max(q_last_stem, q_first_disc) + 1
    if prefill_end > len(ids_fc):
        raise RuntimeError("prefill exceeds the forced sequence")
    input_ids_stem = ids_fc[:prefill_end]

    # ---- image-token spans in prefill coordinates ----
    img_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
    spans: List[Tuple[int, int]] = []
    i, T = 0, len(input_ids_stem)
    while i < T:
        if input_ids_stem[i] == img_id:
            j = i
            while j < T and input_ids_stem[j] == img_id:
                j += 1
            spans.append((i, j - i))
            i = j
        else:
            i += 1
    if len(spans) != len(card_paths):
        raise RuntimeError(
            f"expected {len(card_paths)} image spans, got {len(spans)}"
        )

    return VisualCell(
        state_id=state["state_id"], k=k,
        input_ids_stem=input_ids_stem,
        forced_correct_ids=ids_fc, forced_distractor_ids=ids_fd,
        action_start=action_start, L_prefix=L_prefix, stem_len=stem_len,
        q_last_stem=q_last_stem, q_first_disc=q_first_disc,
        action_tokens_correct=at_c, action_tokens_distractor=at_d,
        n_action_tokens=n_action_tokens,
        card_paths=list(card_paths), image_spans=spans,
        pixel_values=base_inputs["pixel_values"], image_grid_thw=grid_thw,
        stem_prefix_ok_c=stem_prefix_ok_c, stem_prefix_ok_d=stem_prefix_ok_d,
        base_ids_match_processor=base_ids_match,
        debug={
            "merge_size": int(processor.image_processor.merge_size),
            "T_prefill": len(input_ids_stem),
            "T_correct": len(ids_fc),
            "T_distractor": len(ids_fd),
            "q_last_stem_token": tokenizer.decode([input_ids_stem[q_last_stem]]),
            "fix": "li_style_stem_tokenization_visual",
        },
    )


# ---------------------------------------------------------------------------
# region → absolute token positions
# ---------------------------------------------------------------------------
def region_to_absolute(cell: VisualCell, region_meta: dict,
                       key: str) -> List[int]:
    """Map per-card patch indices from a region_patches / patch_indices record
    into absolute prefill token positions.

    `key` is one of "kw_patch_indices", "inj_patch_indices", "fil_patch_indices"
    (region_patches) or "keyword_patch_indices" (patch_indices).

    Asserts the card token counts still match, so a stale region file cannot be
    silently applied to a different render.
    """
    cards = region_meta.get("cards") or []
    if len(cards) != len(cell.image_spans):
        raise RuntimeError(
            f"region file has {len(cards)} cards, cell has {len(cell.image_spans)}"
        )
    out = set()
    for (start, n_tok), card in zip(cell.image_spans, cards):
        if int(card["n_image_tokens"]) != n_tok:
            raise RuntimeError(
                f"card {card.get('card')}: region file says "
                f"{card['n_image_tokens']} tokens, cell has {n_tok}"
            )
        for pi in card.get(key) or []:
            if 0 <= int(pi) < n_tok:
                out.add(start + int(pi))
    return sorted(out)


def all_image_positions(cell: VisualCell) -> List[int]:
    out = set()
    for start, n in cell.image_spans:
        out.update(range(start, start + n))
    return sorted(out)
