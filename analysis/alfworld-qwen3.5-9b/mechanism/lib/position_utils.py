"""
Compute the query position and KW/MC/NON token index sets for one cell.

Pipeline:
1. Build lexical-interference's `messages` via `probe_lib.build_messages`.
2. Render to a string with `apply_chat_template(..., tokenize=False,
   add_generation_prompt=True)`; tokenize with `return_offsets_mapping=True`.
   That gives `base_ids, base_offsets` and `L_prefix = len(base_ids)`.
3. Re-render with the forced-assistant turn appended (continue_final_message);
   tokenize the same way to get `forced_ids` and its offsets.
4. The stem `FORCED_STEM` sits between `L_prefix` and the action tokens.
   We find its exact token count by locating the character offset of the
   *end* of `Action:` in the forced-turn rendered string, then reading off
   the number of forced_ids whose end-offset lies within `[base_text_end,
   stem_end_char]`. This count is off by one (see `_stem_length`); scoring
   uses `lib/text_positions.py` instead.
5. KW/MC positions come from regex over the *skill-body slice* of the base
   rendered string.
"""
from __future__ import annotations
import re
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .prompts_bridge import import_li_prompts, import_li_probe_lib_light


FORCED_THOUGHT = "Thought: I will now perform the action."
FORCED_STEM = f"{FORCED_THOUGHT}\nAction:"


@dataclass
class CellPositions:
    state_id: str
    k: int
    variant_file: str
    input_ids: List[int]
    L_prefix: int
    stem_len: int
    n_action_tokens: int
    q_last_stem: int
    decision_span: Tuple[int, int]
    q_first_disc: int
    kw_positions: List[int]
    mc_positions: List[int]
    non_positions_count: int
    keyword: str
    control_word: Optional[str]
    action_tokens_correct: List[int]
    action_tokens_distractor: List[int]
    stem_tokens: List[int] = field(default_factory=list)
    debug: Dict = field(default_factory=dict)


_TEMPLATE_KWARG_CACHE: Dict[int, bool] = {}


def _template_supports_thinking(tokenizer) -> bool:
    tid = id(tokenizer)
    if tid in _TEMPLATE_KWARG_CACHE:
        return _TEMPLATE_KWARG_CACHE[tid]
    template = getattr(tokenizer, "chat_template", "") or ""
    _TEMPLATE_KWARG_CACHE[tid] = "enable_thinking" in template
    return _TEMPLATE_KWARG_CACHE[tid]


def _apply(tokenizer, messages, *, forced_action: Optional[str] = None) -> str:
    kwargs = {}
    if _template_supports_thinking(tokenizer):
        kwargs["chat_template_kwargs"] = {"enable_thinking": False}
    if forced_action is None:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **kwargs,
        )
    forced = list(messages) + [
        {"role": "assistant",
         "content": f"{FORCED_THOUGHT}\nAction: {forced_action}"}
    ]
    return tokenizer.apply_chat_template(
        forced, tokenize=False, continue_final_message=True,
        add_generation_prompt=False, **kwargs,
    )


def _tokenize(tokenizer, text):
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    return list(enc["input_ids"]), list(enc["offset_mapping"])


def _render(tokenizer, messages, *, forced_action=None):
    """Render, tokenize and return (text, ids, offsets)."""
    text = _apply(tokenizer, messages, forced_action=forced_action)
    ids, offsets = _tokenize(tokenizer, text)
    return text, ids, offsets


def _match_spans(needle: str, hay: str) -> List[Tuple[int, int]]:
    if not needle:
        return []
    pattern = re.compile(rf"\b{re.escape(needle)}\b", re.IGNORECASE)
    return [(m.start(), m.end()) for m in pattern.finditer(hay)]


def _spans_to_token_indices(spans, offsets, body_start, body_end):
    out = []
    for s, e in spans:
        s_abs, e_abs = s + body_start, e + body_start
        if s_abs < body_start or e_abs > body_end:
            continue
        for idx, (a, b) in enumerate(offsets):
            if b <= s_abs or a >= e_abs:
                continue
            out.append(idx)
    return sorted(set(out))


def _stem_length(base_text: str, forced_text: str, forced_offsets) -> int:
    """Count tokens in `forced_ids` that belong to the stem (between base_text
    and 'Action: ' + action).

    Off by one.  Character-offset counting includes the
    token that straddles the base/forced boundary (the empty think-wrapper makes
    base_text end with `<think>\\n` while the forced turn continues
    `\\n</think>\\n\\nThought:...`, so a merged '\\n\\n' token starts before
    base_end_char).  The result is +1 too large, which shifts action_start one
    token right and drops the discriminative verb from the scored span.

    Only `compute_positions` (which raises unless allow_legacy=True) calls it.
    Do not use it to anchor anything.
    """
    marker = "\nAction: "
    marker_idx = forced_text.rfind(marker)
    if marker_idx < 0:
        marker = "Action: "
        marker_idx = forced_text.rfind(marker)
    if marker_idx < 0:
        raise RuntimeError("could not locate 'Action:' in forced-turn text")
    # Stem includes up through the colon after 'Action', i.e. through
    # marker_idx + marker.rstrip(' ').__len__() + 1 (colon).
    stem_end_char = marker_idx + len(marker.rstrip(" "))
    # Count tokens whose end-offset is <= stem_end_char and start-offset >= base_end_char.
    base_end_char = len(base_text)
    count = 0
    for start, end in forced_offsets:
        if end <= base_end_char:
            continue        # part of the base prompt
        if start >= stem_end_char:
            break           # part of the action
        count += 1
    return count


def compute_positions(
    *,
    tokenizer,
    state: dict,
    k: int,
    skill_body: str,
    variant_file: str,
    control_word: Optional[str] = None,
    allow_legacy: bool = False,
) -> CellPositions:
    """LEGACY — superseded by lib/text_positions.py.

    `_stem_length` counts stem tokens by character offsets and includes the
    token straddling the base/forced boundary, so it comes out +1 too large.
    `action_start` then shifts one token right and the FIRST action token — the
    discriminative verb — is dropped from the scored span, which under-measures
    the exposure slope by ~32%.

    It raises by default so no artifact can be silently produced from it. Pass
    `allow_legacy=True`, or set MECH_ALLOW_LEGACY_POSITIONS=1, only to compare
    against the legacy behaviour.

    Use `compute_text_positions` instead.  Its `action_start` comes from the
    stem-only tokenisation and is guarded by `stem_prefix_ok`.
    """
    if not (allow_legacy or os.environ.get("MECH_ALLOW_LEGACY_POSITIONS") == "1"):
        raise RuntimeError(
            "lib.position_utils.compute_positions is the LEGACY position helper and "
            "is off by one in the stem length: action_start "
            "lands one token too far right, dropping the discriminative verb "
            "from the scored span and under-measuring the exposure slope by "
            "~32%. Use lib/text_positions.compute_text_positions "
            "instead (text) or lib/visual_positions.build_visual_cell "
            "(visual). If you deliberately need the legacy behaviour, "
            "pass allow_legacy=True or set "
            "MECH_ALLOW_LEGACY_POSITIONS=1."
        )
    prompts = import_li_prompts()
    probe_lib = import_li_probe_lib_light()

    messages = probe_lib.build_messages(state, "text", skill_text=skill_body)
    base_text = _apply(tokenizer, messages)
    base_ids, base_offsets = _tokenize(tokenizer, base_text)
    L_prefix = len(base_ids)

    # Skill-body slice for KW/MC regex matching.
    intro = prompts.SKILL_INTRO_TEXT.strip()
    outro = prompts.SKILL_OUTRO_TEXT.strip().split("\n")[0]
    intro_pos = base_text.rfind(intro)
    outro_pos = base_text.rfind(outro)
    if intro_pos == -1 or outro_pos == -1 or outro_pos <= intro_pos:
        raise RuntimeError(f"skill-body slice not located for state={state['state_id']} k={k}")
    body_start = intro_pos + len(intro)
    body_end = outro_pos
    body = base_text[body_start:body_end]

    keyword = state["distractor_keyword"]
    kw_spans = _match_spans(keyword, body)
    kw_positions = _spans_to_token_indices(kw_spans, base_offsets, body_start, body_end)

    mc_positions: List[int] = []
    n_mc_matches = 0
    if control_word:
        mc_spans = _match_spans(control_word, body)
        n_mc_matches = len(mc_spans)
        mc_positions = _spans_to_token_indices(mc_spans, base_offsets, body_start, body_end)

    # Forced turns to isolate stem length + action tokens.
    correct = state["correct_action"]
    distractor = state["distractor_action"]
    forced_c_text = _apply(tokenizer, messages, forced_action=correct)
    forced_d_text = _apply(tokenizer, messages, forced_action=distractor)
    forced_c_ids, forced_c_offsets = _tokenize(tokenizer, forced_c_text)
    forced_d_ids, forced_d_offsets = _tokenize(tokenizer, forced_d_text)

    stem_len = _stem_length(base_text, forced_c_text, forced_c_offsets)
    # Sanity: the two forced turns must have the same stem length.
    stem_len_d = _stem_length(base_text, forced_d_text, forced_d_offsets)
    if stem_len != stem_len_d:
        stem_len = min(stem_len, stem_len_d)

    q_last_stem = L_prefix + stem_len - 1
    action_start = L_prefix + stem_len
    action_tokens_correct = forced_c_ids[action_start:]
    action_tokens_distractor = forced_d_ids[action_start:]
    n_action_tokens = max(len(action_tokens_correct), len(action_tokens_distractor))

    q_first_disc = q_last_stem
    for i, (a, b) in enumerate(zip(action_tokens_correct, action_tokens_distractor)):
        if a != b:
            q_first_disc = action_start + i - 1
            break
    else:
        q_first_disc = q_last_stem

    # The prefilled sequence we forward through the model must cover every
    # query position we plan to read attention/logits at — i.e. through
    # q_first_disc inclusive. `forced_c_ids` is used because the two forced
    # sequences share their prefix up to the divergence point.
    prefill_end = max(q_last_stem, q_first_disc) + 1
    prefill_ids = forced_c_ids[:prefill_end]

    non_count = L_prefix - (len(kw_positions) + len(mc_positions))
    return CellPositions(
        state_id=state["state_id"],
        k=k,
        variant_file=variant_file,
        input_ids=prefill_ids,
        L_prefix=L_prefix,
        stem_len=stem_len,
        n_action_tokens=n_action_tokens,
        q_last_stem=q_last_stem,
        decision_span=(q_last_stem, q_last_stem + max(1, n_action_tokens) - 1),
        q_first_disc=q_first_disc,
        kw_positions=kw_positions,
        mc_positions=mc_positions,
        non_positions_count=non_count,
        keyword=keyword,
        control_word=control_word,
        action_tokens_correct=action_tokens_correct,
        action_tokens_distractor=action_tokens_distractor,
        stem_tokens=forced_c_ids[L_prefix:L_prefix + stem_len],
        debug={
            "body_char_len": body_end - body_start,
            "n_kw_matches": len(kw_spans),
            "n_mc_matches": n_mc_matches,
        },
    )


def load_state(state_id: str, states_dir: str) -> dict:
    with open(os.path.join(states_dir, f"{state_id}.json")) as f:
        return json.load(f)
