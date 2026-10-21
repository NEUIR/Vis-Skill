"""
Position computation with stem-only tokenisation.

`lib/position_utils.py::_stem_length` counts stem tokens by *character offsets* in
the combined forced text and includes the token that STRADDLES the base/forced
boundary (the empty think-wrapper makes base_text end with `<think>\n` while the
forced turn continues `\n</think>\n\nThought:...`, so a merged `'\n\n'` token
starts before base_end_char). stem_len comes out +1 too large -> action_start
shifts one token right -> the first action token (the discriminative VERB) is
dropped from the scored span, and q_last_stem / q_first_disc are mis-anchored
by +1.

This module mirrors lexical-interference's probe_lib.score_action in
HF-tokenizer space:
  action_start = len( tokenize( messages + [assistant: FORCED_STEM] ) )
i.e. tokenize the *stem-only* assistant turn as its own sequence and take its
length. Guarded by `stem_prefix_ok` (the stem tokenization must be a literal
prefix of the full forced tokenization). This makes n_action_tokens equal to
lexical-interference's ntok_distractor and keeps the verb in the scored span;
q_last_stem is the true last stem token (':').

Reuses lib.position_utils helpers so the rest of the behaviour (KW/ctrl regex over the
skill-body slice, CellPositions shape) is byte-identical to lib.position_utils.
"""
from __future__ import annotations
import os, sys
from typing import List, Optional

MECH_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if MECH_ROOT not in sys.path:
    sys.path.insert(0, MECH_ROOT)

from lib.position_utils import (                       # noqa: E402
    CellPositions, FORCED_THOUGHT, FORCED_STEM,
    _tokenize, _match_spans, _spans_to_token_indices, _template_supports_thinking,
)
from lib.prompts_bridge import (                  # noqa: E402
    import_li_prompts, import_li_probe_lib_light,
)


def _kwarg(tokenizer):
    if _template_supports_thinking(tokenizer):
        return {"chat_template_kwargs": {"enable_thinking": False}}
    return {}


def _render_assistant(tokenizer, messages, assistant_content: str) -> str:
    """apply_chat_template with a continued assistant turn (continue_final_message)."""
    forced = list(messages) + [{"role": "assistant", "content": assistant_content}]
    return tokenizer.apply_chat_template(
        forced, tokenize=False, continue_final_message=True,
        add_generation_prompt=False, **_kwarg(tokenizer),
    )


def _render_base(tokenizer, messages) -> str:
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **_kwarg(tokenizer),
    )


def compute_text_positions(
    *,
    tokenizer,
    state: dict,
    k: int,
    skill_body: str,
    variant_file: str,
    control_word: Optional[str] = None,
) -> CellPositions:
    """Drop-in replacement for lib.position_utils.compute_positions with the stem fix.

    `control_word`, when given, is located in THIS SAME document (the original),
    yielding `mc_positions` = the matched-word control columns for the within-doc
    DiD / ablation. The CKA control word appears exactly k times in the
    original doc at k in {4, 8}.
    """
    prompts = import_li_prompts()
    probe_lib = import_li_probe_lib_light()

    messages = probe_lib.build_messages(state, "text", skill_text=skill_body)

    # ---- base prompt (add_generation_prompt) -> L_prefix + offsets for regex ----
    base_text = _render_base(tokenizer, messages)
    base_ids, base_offsets = _tokenize(tokenizer, base_text)
    L_prefix = len(base_ids)

    # ---- skill-body slice for KW / control-word matching ----
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

    # ---- stem-only tokenization ----
    stem_text = _render_assistant(tokenizer, messages, FORCED_STEM)   # "...Action:"
    stem_ids, _ = _tokenize(tokenizer, stem_text)
    action_start = len(stem_ids)              # absolute index where action tokens begin
    stem_len = action_start - L_prefix        # stem tokens contributed by the forced turn

    correct = state["correct_action"]
    distractor = state["distractor_action"]
    forced_c_text = _render_assistant(tokenizer, messages, f"{FORCED_THOUGHT}\nAction: {correct}")
    forced_d_text = _render_assistant(tokenizer, messages, f"{FORCED_THOUGHT}\nAction: {distractor}")
    forced_c_ids, _ = _tokenize(tokenizer, forced_c_text)
    forced_d_ids, _ = _tokenize(tokenizer, forced_d_text)

    # LI guard: the stem tokenization must be a literal prefix of each full turn.
    stem_prefix_ok_c = forced_c_ids[:action_start] == stem_ids
    stem_prefix_ok_d = forced_d_ids[:action_start] == stem_ids

    q_last_stem = action_start - 1
    action_tokens_correct = forced_c_ids[action_start:]
    action_tokens_distractor = forced_d_ids[action_start:]
    n_action_tokens = max(len(action_tokens_correct), len(action_tokens_distractor))

    q_first_disc = q_last_stem
    for i, (a, b) in enumerate(zip(action_tokens_correct, action_tokens_distractor)):
        if a != b:
            q_first_disc = action_start + i - 1
            break

    prefill_end = max(q_last_stem, q_first_disc) + 1
    prefill_ids = forced_c_ids[:prefill_end]
    non_count = L_prefix - (len(kw_positions) + len(mc_positions))

    return CellPositions(
        state_id=state["state_id"], k=k, variant_file=variant_file,
        input_ids=prefill_ids, L_prefix=L_prefix, stem_len=stem_len,
        n_action_tokens=n_action_tokens, q_last_stem=q_last_stem,
        decision_span=(q_last_stem, q_last_stem + max(1, n_action_tokens) - 1),
        q_first_disc=q_first_disc, kw_positions=kw_positions, mc_positions=mc_positions,
        non_positions_count=non_count, keyword=keyword, control_word=control_word,
        action_tokens_correct=action_tokens_correct,
        action_tokens_distractor=action_tokens_distractor,
        stem_tokens=stem_ids[L_prefix:],
        debug={
            "body_char_len": body_end - body_start,
            "n_kw_matches": len(kw_spans),
            "n_mc_matches": n_mc_matches,
            "action_start": action_start,
            "stem_prefix_ok_c": stem_prefix_ok_c,
            "stem_prefix_ok_d": stem_prefix_ok_d,
            "fix": "li_style_stem_tokenization",
        },
    )
