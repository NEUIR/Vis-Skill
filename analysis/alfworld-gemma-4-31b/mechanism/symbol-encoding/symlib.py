"""
symbol-encoding shared library — "what is the keyword ENCODED AS?"

The question this folder answers
-------------------------------
Identical skill content, identical keyword count k, only the MODALITY differs,
yet the exposure slope d logP(distractor)/dk is text 0.741 vs visual 0.552
(LI per-state means; on k in {0,4,8} text is 0.846). The hypothesis under test:

  TEXT keyword  = a NATIVE DISCRETE TOKEN. " cabinet" is literally one symbol in
                  the vocabulary, so k repetitions = k crisp symbols landing
                  directly in the LM's lexical/associative machinery.
  VISUAL keyword = PIXELS that must be re-recognised through the vision tower ->
                  2x2 patch merge -> projector. Lossy. k repetitions = k blurry
                  regions (and growing AREA), not k crisp symbols.

The decisive manipulation (this module's core)
----------------------------------------------
CASING. Rewriting `cabinet` -> `CABINET` -> `CaBiNeT` holds constant: the
characters, the word, the semantics, the human legibility, the word count, the
slot positions, the value of k, and the scored action string. It changes ONE
thing in the text condition: BPE segmentation.

    ' cabinet'   -> 1 token     ' microwave' -> 1 token
    ' Cabinet'   -> 1 token     ' Microwave' -> 1 token
    ' CABINET'   -> 2 tokens    ' MICROWAVE' -> 3 tokens
    ' CaBiNeT'   -> 4 tokens    ' MiCrOwAvE' -> 6 tokens

Rendered to a PNG, the very same strings change segmentation NOT AT ALL — pixels
have no BPE. So the crossed design

              text            visual
  native      steep    ?      steep    ?
  altcase     SHALLOW  ?      steep    ?

has a signature that only the "native discrete token" account predicts: a
modality x casing INTERACTION. If instead the drop appears in both modalities it
is surface-form mismatch, not tokenisation; if it appears in neither, symbol
crispness is not what carries the exposure effect.

`title` is the specificity control on the text side: it perturbs the surface form
just like `upper` but (for cabinet/safe/drawer/microwave) stays 1 token. A drop at
`upper`/`altcase` but not at `title` isolates segmentation from surface mismatch.
`ctrl_altcase` is the second specificity control: mangle a FREQUENCY-MATCHED
non-keyword content word instead, keyword left native — mangling *any* word must
not move the slope.

Scoring conventions
-------------------
- action_start = len(tokenize(render(messages + [assistant: FORCED_STEM]))).
  The character-offset method in lib/position_utils.py is off by one and silently
  drops the first (discriminative verb) action token. We use the LI-style
  stem-only tokenisation and assert `stem_prefix_ok` on every cell.
- The scored span is ALL action tokens from action_start onward, for the
  distractor and the correct action separately.
- Effect sizes are per-state PAIRED with bootstrap CIs and clustered by game;
  p-values are secondary. See analyze_*.py.
"""
from __future__ import annotations

import os
import re
import sys
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
MECH_ROOT = os.path.dirname(HERE)
if MECH_ROOT not in sys.path:
    sys.path.insert(0, MECH_ROOT)

from lib.config import FORCED_THOUGHT, LI_VARIANTS_DIR, MODEL_PATH  # noqa: E402
from lib.prompts_bridge import import_li_probe_lib_light  # noqa: E402

FORCED_STEM = f"{FORCED_THOUGHT}\nAction:"
IMAGE_TOKEN = "<|image_pad|>"
VISION_BLOCK = "<|vision_start|>" + IMAGE_TOKEN + "<|vision_end|>"

# k levels. {0,4,8} is the slope convention (text baseline 0.846) and is
# what makes this fast; k=0 has no keyword at all so casing is undefined there
# and the k=0 cell is SHARED across casing arms.
KS = [0, 4, 8]


# ════════════════════════════════════════════════════════════════════════
# Casing transforms — the manipulation
# ════════════════════════════════════════════════════════════════════════

def _altcase(w: str) -> str:
    return "".join(c.upper() if i % 2 == 0 else c.lower() for i, c in enumerate(w))


CASINGS: Dict[str, callable] = {
    "native": lambda w: w,
    "title": lambda w: w.capitalize(),
    "upper": lambda w: w.upper(),
    "altcase": _altcase,
}

# Arms scored at k>0. `ctrl_altcase` mangles a frequency-matched control word and
# leaves the keyword native (specificity control).
ARMS = ["native", "title", "upper", "altcase", "ctrl_altcase"]


def recase_word_in_doc(doc: str, word: str, casing: str) -> Tuple[str, int]:
    """Rewrite every whole-word occurrence of `word` under `casing`.

    Returns (new_doc, n_replaced). Whole-word + case-insensitive, matching the
    regex convention used everywhere else in this analysis tree
    (lib/position_utils._match_spans). The keyword occurs exactly k times in the LI
    original documents (165/165 at k=4, 128/128 at k=8) and ALWAYS inside
    an injected sentence, so this touches exactly the k injected occurrences and
    nothing in the base skill prose.
    """
    fn = CASINGS[casing]
    n = 0

    def _rep(m):
        nonlocal n
        n += 1
        return fn(m.group(0))

    return re.sub(rf"\b{re.escape(word)}\b", _rep, doc, flags=re.I), n


# ════════════════════════════════════════════════════════════════════════
# Frequency-matched control word (for the ctrl_altcase arm)
# ════════════════════════════════════════════════════════════════════════

_STOP = set("""the a an and or but if then else when while for of to in on at by with from as is are was
were be been being do does did have has had will would shall should can could may might must not no yes
this that these those it its they them their there here you your we our i me my he she his her him
each every all any both few more most other some such only own same so than too very s t don now
use used using step steps first second third next before after during into over under out up down off""".split())


def pick_control_word(doc: str, keyword: str, k: int, avoid: List[str]) -> Optional[str]:
    """A non-keyword content word occurring EXACTLY k times (whole-word) in `doc`.

    Frequency-matched to the keyword by construction. Among candidates we take
    the one closest in character length to the keyword (so the casing mangle
    perturbs a comparable amount of surface), tie-broken alphabetically so the
    choice is deterministic and seed-free.

    `avoid` holds words that must not be touched — anything appearing in the
    correct or distractor action string, since recasing those would change the
    literal overlap with the SCORED action and confound the arm.
    """
    avoid_l = {w.lower() for w in avoid} | {keyword.lower()} | _STOP
    counts: Dict[str, int] = {}
    for m in re.finditer(r"[A-Za-z][a-z]{4,}", doc):
        w = m.group(0)
        wl = w.lower()
        if wl in avoid_l:
            continue
        counts[wl] = counts.get(wl, 0) + 1
    # re-verify with the same whole-word regex used for replacement
    cands = []
    for wl, c in counts.items():
        if c != k:
            continue
        if len(re.findall(rf"\b{re.escape(wl)}\b", doc, flags=re.I)) != k:
            continue
        cands.append(wl)
    if not cands:
        return None
    cands.sort(key=lambda w: (abs(len(w) - len(keyword)), w))
    return cands[0]


# ════════════════════════════════════════════════════════════════════════
# Token-count measurement — the REGRESSOR, not the nominal condition
# ════════════════════════════════════════════════════════════════════════

def keyword_token_count(tok, word: str) -> int:
    """BPE tokens for the keyword as it appears mid-sentence (leading space).

    Nominal casing is NOT the predictor: `safe` is 1 token even in UPPER, while
    `microwave` goes 1 -> 3 -> 6. Analyses regress on this measured count.
    """
    return len(tok(" " + word, add_special_tokens=False)["input_ids"])


# ════════════════════════════════════════════════════════════════════════
# Chat-template plumbing
# ════════════════════════════════════════════════════════════════════════

_TM_CACHE: Dict[int, dict] = {}


def tm_kwargs(tok):
    key = id(tok)
    if key not in _TM_CACHE:
        t = getattr(tok, "chat_template", "") or ""
        _TM_CACHE[key] = (
            {"chat_template_kwargs": {"enable_thinking": False}} if "enable_thinking" in t else {}
        )
    return _TM_CACHE[key]


def render_chat(tok_or_proc, messages, *, forced_action=None, stem_only=False) -> str:
    """apply_chat_template -> str. Three modes used by the scorers.

    - stem_only=True     : messages + assistant "Thought: ...\\nAction:"  (continued)
    - forced_action=str  : messages + assistant "Thought: ...\\nAction: <act>"
    - neither            : messages + generation prompt
    """
    tk = getattr(tok_or_proc, "tokenizer", tok_or_proc)
    kw = tm_kwargs(tk)
    if stem_only:
        seq = list(messages) + [{"role": "assistant", "content": FORCED_STEM}]
        return tok_or_proc.apply_chat_template(
            seq, tokenize=False, continue_final_message=True,
            add_generation_prompt=False, **kw)
    if forced_action is not None:
        seq = list(messages) + [
            {"role": "assistant", "content": f"{FORCED_THOUGHT}\nAction: {forced_action}"}]
        return tok_or_proc.apply_chat_template(
            seq, tokenize=False, continue_final_message=True,
            add_generation_prompt=False, **kw)
    return tok_or_proc.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **kw)


def ids_of(tok, text: str) -> List[int]:
    return list(tok(text, add_special_tokens=False)["input_ids"])


def expand_image_pads(rendered: str, grid_thw, merge_len: int) -> str:
    """Replace each single <|image_pad|> placeholder with its real pad run.

    The chat template emits one placeholder per image; the processor normally
    expands it. We tokenise text-only (so we can compute action_start in exact
    token space) and therefore must expand by hand, using the same
    t*h*w/merge_len arithmetic the processor uses.
    """
    parts = rendered.split(VISION_BLOCK)
    if len(parts) - 1 != len(grid_thw):
        raise RuntimeError(f"{len(parts) - 1} vision blocks vs {len(grid_thw)} grids")
    out = parts[0]
    for i, thw in enumerate(grid_thw):
        t, h, w = (int(x) for x in thw)
        out += ("<|vision_start|>" + IMAGE_TOKEN * (t * h * w // merge_len)
                + "<|vision_end|>" + parts[i + 1])
    return out


# ════════════════════════════════════════════════════════════════════════
# Scoring — the LI-correct forced-decoding score
# ════════════════════════════════════════════════════════════════════════

def score_actions_text(model, tok, messages, correct: str, distractor: str, device) -> dict:
    """Sum log P over ALL action tokens, for both actions, in the text condition.

    action_start comes from the stem-only tokenisation. We assert
    the stem is a literal token prefix of each forced sequence, which is what
    guarantees the discriminative verb token is inside the scored span.
    """
    import torch

    stem_ids = ids_of(tok, render_chat(tok, messages, stem_only=True))
    action_start = len(stem_ids)

    out = {"action_start": action_start, "stem_prefix_ok": True}
    for name, act in (("distractor", distractor), ("correct", correct)):
        ids = ids_of(tok, render_chat(tok, messages, forced_action=act))
        if ids[:action_start] != stem_ids:
            out["stem_prefix_ok"] = False
        with torch.inference_mode():
            logits = model(
                input_ids=torch.tensor([ids], device=device),
                attention_mask=torch.ones(1, len(ids), dtype=torch.long, device=device),
                use_cache=False, return_dict=True).logits[0]
            # Same fp32-cast-after-slicing fix as the visual scorer; per-row
            # log_softmax is unaffected by dropping other rows, so this is
            # numerically identical to casting the whole tensor, just far cheaper.
            sl = logits[action_start - 1:len(ids) - 1].float()
            lp = torch.log_softmax(sl, dim=-1)
            per = [float(lp[i - action_start, ids[i]]) for i in range(action_start, len(ids))]
            del logits, sl, lp
        out[f"logp_{name}"] = float(sum(per))
        out[f"n_{name}"] = len(per)
        out[f"per_token_{name}"] = per
        out[f"T_{name}"] = len(ids)
    return out


def score_actions_visual(model, tok, processor, messages, images,
                         correct: str, distractor: str, device) -> dict:
    """Visual counterpart. Pixels come from `images` (may be blurred/re-rendered
    in memory); the text scaffold comes from `messages`."""
    import torch

    base = processor(text=[render_chat(processor, messages)], images=images,
                     return_tensors="pt")
    grid_thw = base["image_grid_thw"]
    pv = base["pixel_values"].to(device=device, dtype=torch.bfloat16)
    gthw = grid_thw.to(device=device)
    merge_len = int(processor.image_processor.merge_size) ** 2
    grid_list = grid_thw.tolist()

    stem_ids = ids_of(tok, expand_image_pads(
        render_chat(processor, messages, stem_only=True), grid_list, merge_len))
    action_start = len(stem_ids)

    out = {"action_start": action_start, "stem_prefix_ok": True,
           "n_image_tokens": int(sum(t * h * w // merge_len for t, h, w in grid_list))}
    for name, act in (("distractor", distractor), ("correct", correct)):
        ids = ids_of(tok, expand_image_pads(
            render_chat(processor, messages, forced_action=act), grid_list, merge_len))
        if ids[:action_start] != stem_ids:
            out["stem_prefix_ok"] = False
        with torch.inference_mode():
            logits = model(
                input_ids=torch.tensor([ids], device=device),
                attention_mask=torch.ones(1, len(ids), dtype=torch.long, device=device),
                pixel_values=pv, image_grid_thw=gthw,
                use_cache=False, return_dict=True).logits[0]
            # Slice to the SCORED ROWS BEFORE the fp32 cast. Casting the whole
            # [T, vocab] tensor costs T*151k*4 bytes, which is ~15 GiB on the long
            # image-pad prompts and OOMs an 80 GiB card. Row i-1 predicts token i,
            # so rows [action_start-1, len(ids)-1) are exactly what we need.
            sl = logits[action_start - 1:len(ids) - 1].float()
            lp = torch.log_softmax(sl, dim=-1)
            per = [float(lp[i - action_start, ids[i]]) for i in range(action_start, len(ids))]
            del logits, sl, lp
        out[f"logp_{name}"] = float(sum(per))
        out[f"n_{name}"] = len(per)
        out[f"per_token_{name}"] = per
        out[f"T_{name}"] = len(ids)
    return out


# ════════════════════════════════════════════════════════════════════════
# Model loading
# ════════════════════════════════════════════════════════════════════════

def load_text_model(device):
    """Text-only path still loads the full multimodal checkpoint (same weights,
    same numerics as the visual arm — required for text/visual comparability)."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="sdpa", device_map={"": device})
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def load_visual_model(device):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="sdpa", device_map={"": device})
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, processor, tok


# ════════════════════════════════════════════════════════════════════════
# Cells
# ════════════════════════════════════════════════════════════════════════

def text_doc_path(state_id: str, k: int) -> str:
    return os.path.join(LI_VARIANTS_DIR, "text", state_id, f"level_{k}.md")


def build_messages_text(state, doc: str):
    return import_li_probe_lib_light().build_messages(state, "text", skill_text=doc)


def build_messages_visual(state, card_paths):
    return import_li_probe_lib_light().build_messages(state, "visual", card_paths=card_paths)


def cell_id(state_id: str, k: int, arm: str) -> str:
    return f"{state_id}__k{k}__{arm}"


def plan_cells(state_ids: List[str], ks: List[int], arms: List[str]):
    """(state_id, k, arm) triples. k=0 carries no keyword so it is scored once,
    under `native`, and shared by every arm at analysis time."""
    for sid in state_ids:
        for k in ks:
            if k == 0:
                yield sid, 0, "native"
            else:
                for arm in arms:
                    yield sid, k, arm
