"""
Inference primitives for the lexical-interference probe.

Two measurements per (state × k × representation) cell:

1. **Free generation** — what the model actually does (temperature 0), which is
   what produces the "final wrong behaviour" end of the chain.

2. **Forced decoding over the whole candidate action set** — the summed
   log-probability of every currently-executable action under a byte-identical
   prefix, which turns into a normalised action distribution and therefore into
       P(correct), P(distractor), H = -Σ p log p.

The forced-decoding path uses vLLM's chat API with
`continue_final_message=True` + `prompt_logprobs`, which is the only variant that
keeps multimodal (image) context intact — a plain `/v1/completions` call cannot
carry the rendered skill cards. Token boundaries come from `/tokenize`:

    prefix_count = |tokenize(messages, add_generation_prompt=True)|
    full_tokens  = tokenize(messages + [forced assistant turn],
                            continue_final_message=True)
    logprob(action) = Σ prompt_logprobs[i][full_tokens[i]]
                      for i in [prefix_count, |full_tokens|)

`prefix_count` depends only on the cell, so it is computed once and reused for
every candidate of that cell.
"""
import os
import math
import base64
import requests
from openai import OpenAI
from retry import retry

from prompts import (SYSTEM_PROMPT, SKILL_INTRO_TEXT, SKILL_INTRO_TEXT_VISUAL,
                     SKILL_OUTRO_TEXT, FORCED_THOUGHT)
import skill_lexicon as lex

DEFAULT_MODEL = os.path.join(os.environ.get("MODELS_DIR", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "models"))), "Qwen3.5-9B")
HTTP_TIMEOUT = 300

# ── endpoint management (per worker process) ──
_port = 8000
_clients = {}


def set_endpoint(port):
    global _port
    _port = int(port)


def _client():
    c = _clients.get(_port)
    if c is None:
        c = OpenAI(api_key="EMPTY", base_url=f"http://localhost:{_port}/v1")
        _clients[_port] = c
    return c


def _base():
    return f"http://localhost:{_port}"


# ══════════════════════════════════════════════════════════════════════
# Prompt assembly
# ══════════════════════════════════════════════════════════════════════

def _b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_messages(state, representation, skill_text=None, card_paths=None):
    """system + clean history + one skill-reference turn.

    `representation`:
      text   — the document as characters
      visual — the same document as rendered skill cards
      none   — no skill turn at all (control anchor)

    The wrapper text around the skill body is identical in the text and visual
    conditions; only the modality of the body differs.
    """
    messages = [dict(m) for m in state["history_messages"]]
    if messages and messages[0].get("role") != "system":
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

    if representation == "none":
        # Control anchor: identical decision instruction, but no skill reference
        # exists above — so the wording must not claim one does.
        messages.append({
            "role": "user",
            "content": "Based on the current situation, decide your next action. "
                       "Reply with exactly one action in the required format.",
        })
        return messages

    if representation == "text":
        messages.append({
            "role": "user",
            "content": f"{SKILL_INTRO_TEXT}\n{skill_text}{SKILL_OUTRO_TEXT}",
        })
        return messages

    if representation == "visual":
        content = [{"type": "text", "text": SKILL_INTRO_TEXT_VISUAL}]
        for path in (card_paths or []):
            name = os.path.basename(path).split("_", 1)[-1].rsplit(".", 1)[0]
            content.append({"type": "text", "text": f"\n[Skill: {name}]"})
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{_b64(path)}"}})
        content.append({"type": "text", "text": SKILL_OUTRO_TEXT})
        messages.append({"role": "user", "content": content})
        return messages

    raise ValueError(f"unknown representation: {representation}")


def forced_text(action, template="thought_action"):
    if template == "action_only":
        return f"Action: {action}"
    return f"{FORCED_THOUGHT}\nAction: {action}"


def forced_stem(template="thought_action"):
    """The part of the forced turn that precedes the action string. Constant
    across candidates, so subtracting its log-probability isolates the action
    tokens — which is what makes log P(action) comparable ACROSS k and across
    representations (the stem's own likelihood shifts with the prompt)."""
    return "Action:" if template == "action_only" else f"{FORCED_THOUGHT}\nAction:"


# ══════════════════════════════════════════════════════════════════════
# Free generation
# ══════════════════════════════════════════════════════════════════════

@retry(tries=5, delay=5, backoff=2, jitter=(1, 3))
def generate_action(messages, model=DEFAULT_MODEL, max_tokens=256, top_logprobs=10):
    """Greedy next-turn generation with token logprobs."""
    resp = _client().chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=max_tokens,
        logprobs=True,
        top_logprobs=top_logprobs,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    choice = resp.choices[0]
    text = choice.message.content or ""
    token_lps = None
    if choice.logprobs and choice.logprobs.content:
        token_lps = [{"token": tp.token, "logprob": tp.logprob}
                     for tp in choice.logprobs.content]
    return text, token_lps


# ══════════════════════════════════════════════════════════════════════
# Forced decoding
# ══════════════════════════════════════════════════════════════════════

@retry(tries=5, delay=5, backoff=2, jitter=(1, 3))
def prefix_token_count(messages, model=DEFAULT_MODEL):
    r = requests.post(f"{_base()}/tokenize", json={
        "model": model,
        "messages": messages,
        "add_generation_prompt": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()["count"]


@retry(tries=5, delay=5, backoff=2, jitter=(1, 3))
def _tokenize_forced(messages, model=DEFAULT_MODEL):
    r = requests.post(f"{_base()}/tokenize", json={
        "model": model,
        "messages": messages,
        "continue_final_message": True,
        "add_generation_prompt": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()["tokens"]


@retry(tries=5, delay=5, backoff=2, jitter=(1, 3))
def _prompt_logprobs(messages, model=DEFAULT_MODEL):
    r = requests.post(f"{_base()}/v1/chat/completions", json={
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 1,
        "logprobs": False,
        "prompt_logprobs": 1,
        "continue_final_message": True,
        "add_generation_prompt": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json().get("prompt_logprobs")


def score_action(messages, action, model=DEFAULT_MODEL, prefix_count=None,
                 template="thought_action", stem_tokens=None):
    """-> per-candidate score dict.

    Keys:
      sum_logprob    Σ log p over the whole forced turn (stem + action)
      action_logprob Σ log p over the action tokens only (None if the stem
                     tokenisation is not a prefix of the full tokenisation)
      n_tokens / n_action_tokens
      complete       False when any target token's log-probability could not be
                     read back (tokenizer boundary vs prompt_logprobs mismatch);
                     such a score must not be used.
    """
    forced = list(messages) + [{"role": "assistant",
                                "content": forced_text(action, template)}]
    if prefix_count is None:
        prefix_count = prefix_token_count(messages, model)
    full_tokens = _tokenize_forced(forced, model)
    plps = _prompt_logprobs(forced, model)
    empty = {"sum_logprob": None, "action_logprob": None, "n_tokens": 0,
             "n_action_tokens": 0, "matched_tokens": 0, "complete": False,
             "stem_prefix_ok": None}
    if not plps:
        return empty

    action_start = None
    stem_ok = None
    if stem_tokens:
        stem_ok = list(full_tokens[:len(stem_tokens)]) == list(stem_tokens)
        if stem_ok:
            action_start = len(stem_tokens)

    # Alignment guard: prompt_logprobs must cover exactly the tokenisation that
    # /tokenize produced for the same messages. If the two ever disagree (chat
    # template kwargs ignored, different template version, ...), summing from
    # prefix_count would silently attribute the wrong positions — flag it as
    # incomplete instead. (This is the bug class that invalidated the first
    # visual run of an earlier pilot analysis.)
    length_ok = (len(plps) == len(full_tokens))

    total, action_total = 0.0, 0.0
    matched, n_target, n_action = 0, 0, 0
    for i in range(prefix_count, min(len(plps), len(full_tokens))):
        n_target += 1
        entry = plps[i]
        if not entry:
            continue
        key = str(full_tokens[i])
        if key not in entry:
            continue
        lp = entry[key]["logprob"]
        total += lp
        matched += 1
        if action_start is not None and i >= action_start:
            action_total += lp
            n_action += 1
    complete = bool(n_target) and matched == n_target and length_ok
    return {
        "sum_logprob": total if matched else None,
        "action_logprob": action_total if (action_start is not None and n_action) else None,
        "n_tokens": n_target,
        "n_action_tokens": n_action,
        "matched_tokens": matched,
        "complete": complete,
        "stem_prefix_ok": stem_ok,
        "tokenization_length_ok": length_ok,
    }


def score_candidates(messages, candidates, model=DEFAULT_MODEL, template="thought_action"):
    """Score every candidate action under one shared prefix.

    -> (scores dict {action: score dict}, prefix_count)
    """
    prefix_count = prefix_token_count(messages, model)
    stem_msgs = list(messages) + [{"role": "assistant", "content": forced_stem(template)}]
    try:
        stem_tokens = _tokenize_forced(stem_msgs, model)
    except Exception:
        stem_tokens = None
    scores = {}
    for a in candidates:
        scores[a] = score_action(messages, a, model=model, prefix_count=prefix_count,
                                 template=template, stem_tokens=stem_tokens)
    return scores, prefix_count


# ══════════════════════════════════════════════════════════════════════
# Distribution statistics
# ══════════════════════════════════════════════════════════════════════

def softmax(logits):
    if not logits:
        return []
    m = max(logits)
    exps = [math.exp(l - m) for l in logits]
    z = sum(exps)
    return [e / z for e in exps]


def entropy(probs):
    return float(-sum(p * math.log(p) for p in probs if p > 0))


def distribution_stats(scores, correct_action, distractor_action):
    """Turn per-candidate log-probabilities into the reported quantities.

    Two normalisations are reported:
      sum  — softmax over the summed sequence log-probabilities (the standard
             "renormalise over the candidate set" quantity);
      mean — softmax over length-normalised (per-token) log-probabilities, which
             removes the bias toward short action strings.
    """
    usable = {a: s for a, s in scores.items()
              if s.get("complete") and s.get("sum_logprob") is not None}
    out = {
        "n_scored": len(usable),
        "n_candidates": len(scores),
        "incomplete_candidates": [a for a in scores if a not in usable],
    }
    if not usable:
        return out

    actions = sorted(usable)
    sums = [usable[a]["sum_logprob"] for a in actions]
    # Length normalisation uses the action tokens only, so the per-token average
    # is not diluted by the constant stem.
    means = []
    for a in actions:
        s = usable[a]
        if s.get("action_logprob") is not None and s.get("n_action_tokens"):
            means.append(s["action_logprob"] / s["n_action_tokens"])
        else:
            means.append(s["sum_logprob"] / max(1, s["n_tokens"]))

    for tag, logits in (("sum", sums), ("mean", means)):
        probs = softmax(logits)
        pmap = dict(zip(actions, probs))
        order = sorted(actions, key=lambda a: pmap[a], reverse=True)
        rank = {a: i + 1 for i, a in enumerate(order)}
        h = entropy(probs)
        out[f"p_{tag}"] = pmap
        out[f"entropy_{tag}"] = h
        out[f"entropy_norm_{tag}"] = h / math.log(len(actions)) if len(actions) > 1 else 0.0
        out[f"argmax_{tag}"] = order[0]
        out[f"p_correct_{tag}"] = pmap.get(correct_action)
        out[f"p_distractor_{tag}"] = pmap.get(distractor_action)
        out[f"rank_correct_{tag}"] = rank.get(correct_action)
        out[f"rank_distractor_{tag}"] = rank.get(distractor_action)

    # Absolute log-probabilities. `*_action` excludes the constant forced stem and
    # is therefore the quantity that stays comparable across k and across
    # representations; `logp_*` (stem included) is kept for continuity with
    # an earlier pilot analysis, whose numbers include the stem.
    for tag, key in (("", "sum_logprob"), ("_action", "action_logprob")):
        c = usable.get(correct_action, {}).get(key)
        d = usable.get(distractor_action, {}).get(key)
        out[f"logp_correct{tag}"] = c
        out[f"logp_distractor{tag}"] = d
        out[f"logp_margin{tag}"] = (c - d) if (c is not None and d is not None) else None
    out["ntok_correct"] = usable.get(correct_action, {}).get("n_action_tokens")
    out["ntok_distractor"] = usable.get(distractor_action, {}).get("n_action_tokens")
    out["stem_prefix_ok"] = all(
        s.get("stem_prefix_ok") is not False for s in usable.values())
    return out


# ══════════════════════════════════════════════════════════════════════
# Free-generation bookkeeping
# ══════════════════════════════════════════════════════════════════════

def summarise_generation(text, correct_action, distractor_action, candidates):
    action = lex.parse_action_from_response(text)
    norm = lex.normalize_action(action)
    return {
        "response_text": text,
        "chosen_action_raw": action,
        "chosen_action": norm,
        "chose_correct": norm == lex.normalize_action(correct_action),
        "chose_distractor": norm == lex.normalize_action(distractor_action),
        "chosen_in_candidates": norm in [lex.normalize_action(c) for c in candidates],
        "chosen_verb": lex.action_verb(norm),
    }
