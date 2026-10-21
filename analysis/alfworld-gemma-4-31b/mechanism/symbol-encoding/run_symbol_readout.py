#!/usr/bin/env python3
"""
symbol-encoding / F4 — SYMBOL READOUT: measure directly how available the keyword SYMBOL is
on the language side, text vs visual, as a function of k.

The hypothesis under test says the two modalities differ in what the keyword is
encoded AS: a native discrete token in text, versus pixels the model must
re-recognise in visual. F1/F2b/F3 test that behaviourally (casing). F4 measures
the encoded symbol itself, and does it in a way that is FAIR ACROSS MODALITIES:

  * the probe question is the SAME text string in both arms, appended to the last
    USER turn, so it occupies text token positions in both cases and the scored
    word sits at the start of the assistant turn in both cases;
  * the scored continuation is the SAME native lowercase keyword in both arms;
  * the readout is a MARGIN, logP(keyword) - logP(control word), with the control
    being the frequency-matched non-keyword content word from F1. The margin
    cancels prompt-length and prompt-style effects that necessarily differ
    between a ~4.8k-token text prompt and an image-pad prompt, and which would
    otherwise masquerade as a modality difference in symbol availability.

DV: readout margin R(k) and its slope over k in {0, 4, 8}.
    text slope >> visual slope  -> the symbol IS less available on the language
        side when it arrived as pixels; that is the encoding difference, measured.
    text ~ visual               -> the symbol is equally available and the
        behavioural gap is NOT about symbol availability but about how the action
        computation USES it. That would redirect the story, so it is worth the GPU.

At k=0 the keyword is absent from the document, so there is no frequency-matched
control and `margin` is None; the k=0 cell still records logp_keyword, which is
the no-exposure baseline the slope is anchored on. The margin slope is therefore
fitted on k in {4,8} plus the k=0 keyword-only baseline, and both variants
(margin slope on {4,8}, keyword-logP slope on {0,4,8}) are reported by the
analysis so neither choice has to be defended alone.

MEDIATION (reported, never claimed causal): across states within a modality,
correlate the readout slope with the action-logP exposure slope from F1/F2b.

NOT DONE HERE, deliberately: no teacher-forced prefix of an answer the model has
not produced. The probe scores exactly one candidate word at the first assistant
position, at prefill. Nothing is generated.

Run:
  MECH_CUDA_INDEX=0 python3 run_symbol_readout.py --modality text
  MECH_CUDA_INDEX=1 python3 run_symbol_readout.py --modality visual
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
import traceback

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
MECH_ROOT = os.path.dirname(HERE)
sys.path.insert(0, MECH_ROOT)
sys.path.insert(0, HERE)

from lib.cells import load_manifest, load_state  # noqa: E402
import symlib as S  # noqa: E402
# --- Gemma-4 override (see the same block in run_visual_casing.py) ----------
# symlib.py here is identical to the Qwen implementation; its
# score_actions_visual dies on Gemma-4 with KeyError: 'image_grid_thw'.
import symlib_gemma  # noqa: E402
symlib_gemma.patch(S)
# ----------------------------------------------------------------------------
from run_visual_casing import card_paths_for  # symbol-encoding-rendered cards (native arm)

OUT = os.path.join(HERE, "results", "symbol_readout")
os.makedirs(OUT, exist_ok=True)

# One fixed, modality-neutral question. Identical bytes in both arms.
QUERY = ("\n\nIgnoring the task for a moment: which single object name appears "
         "most often in the skill documents above? Answer with just that one word, "
         "lowercase.")

# The chat template opens a thinking block, so the position right after the
# generation prompt is the first REASONING token, not the answer. We therefore
# use a fixed assistant stem, exactly as the rest of this project does with
# FORCED_STEM ("Thought: ...\nAction:"), so the scored position is the answer
# slot. The stem is identical across k, across arms and across modalities, and
# contains no information about which word the answer is -- it is scaffolding,
# not a teacher-forced prefix of the answer.
READOUT_STEM = "The word is:"


def render_readout(tok_or_proc, messages, stem: str = READOUT_STEM) -> str:
    tk = getattr(tok_or_proc, "tokenizer", tok_or_proc)
    return tok_or_proc.apply_chat_template(
        list(messages) + [{"role": "assistant", "content": stem}],
        tokenize=False, continue_final_message=True, add_generation_prompt=False,
        **S.tm_kwargs(tk))


def append_query(messages, query: str):
    """Append `query` to the last user turn, handling str and content-list forms."""
    msgs = [dict(m) for m in messages]
    for m in reversed(msgs):
        if m.get("role") != "user":
            continue
        c = m["content"]
        if isinstance(c, str):
            m["content"] = c + query
        elif isinstance(c, list):
            c = list(c)
            if c and isinstance(c[-1], dict) and c[-1].get("type") == "text":
                c[-1] = {**c[-1], "text": c[-1]["text"] + query}
            else:
                c.append({"type": "text", "text": query})
            m["content"] = c
        else:
            raise RuntimeError(f"unexpected content type {type(c)}")
        return msgs
    raise RuntimeError("no user turn to append the probe query to")


@torch.no_grad()
def score_words(model, tok, ids, words, device, *, model_kwargs=None):
    """Summed logP of each word as the continuation at the first assistant position.

    The prompt is prefilled ONCE; every candidate is scored against that same
    prefill, so candidates are strictly comparable (same context, same backend,
    same precision). Multi-token words are scored by re-running the prompt with
    the word appended -- deliberately NOT by mutating a shared KV cache, which
    would be prone to cache-reuse bugs.
    """
    mk = {k: v for k, v in (model_kwargs or {}).items()}
    base = torch.tensor([ids], device=device)
    out = model(input_ids=base,
                attention_mask=torch.ones_like(base),
                use_cache=False, return_dict=True,
                **symlib_gemma.align_mm_token_type_ids(mk, len(ids)))
    lp0 = torch.log_softmax(out.logits[0, -1].float(), -1)

    res = {}
    for w in words:
        w_ids = S.ids_of(tok, " " + w)
        if len(w_ids) == 1:
            res[w] = {"logp": float(lp0[w_ids[0]]), "n_tokens": 1,
                      "per_token": [float(lp0[w_ids[0]])]}
            continue
        full = ids + w_ids
        t = torch.tensor([full], device=device)
        o = model(input_ids=t, attention_mask=torch.ones_like(t),
                  use_cache=False, return_dict=True,
                  **symlib_gemma.align_mm_token_type_ids(mk, len(full)))
        # Only the rows that actually score the word are needed: row
        # len(ids)+j-1 predicts w_ids[j]. log_softmax is row-wise, so slicing
        # first is numerically IDENTICAL to log_softmax over the whole
        # [T, vocab] block -- but the full block is T x 262k x 4 bytes in fp32
        # (>20 GiB on a 20k-token visual prompt). Slice, then upcast.
        lo = len(ids) - 1
        rows = o.logits[0, lo:lo + len(w_ids)]          # [n_w, vocab]
        lp = torch.log_softmax(rows.float(), -1)
        per = [float(lp[j, w_ids[j]]) for j in range(len(w_ids))]
        res[w] = {"logp": float(sum(per)), "n_tokens": len(w_ids), "per_token": per}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", choices=["text", "visual"], required=True)
    ap.add_argument("--ks", nargs="*", type=int, default=S.KS)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--states", nargs="*", default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    mod = args.modality
    sids = [s["state_id"] for s in load_manifest()]
    if args.states:
        sids = [s for s in sids if s in args.states]
    if args.limit:
        sids = sids[:args.limit]
    sids = [s for i, s in enumerate(sids) if i % args.num_shards == args.shard]
    cells = [(sid, k) for sid in sids for k in args.ks]
    print(f"[F4/{mod}] states={len(sids)} cells={len(cells)}", flush=True)

    device = f"cuda:{os.environ.get('MECH_CUDA_INDEX', '0')}"
    if mod == "text":
        model, tok = S.load_text_model(device)
        processor = None
    else:
        model, processor, tok = S.load_visual_model(device)
        # Match the end-to-end Gemma baseline (../vis-skill ran at 1120).  The
        # processor default is 280, which renders the dense skill cards at
        # ~254 tokens/image instead of ~1075 -- a different visual condition.
        _mst = symlib_gemma.pin_soft_tokens(processor)
        print(f"[F4] pinned max_soft_tokens={_mst}", flush=True)
    print(f"[F4/{mod}] model ready", flush=True)

    t0 = time.time()
    ndone = nskip = nerr = 0
    for i, (sid, k) in enumerate(cells):
        path = os.path.join(OUT, f"{sid}__k{k}__{mod}.json")
        if os.path.exists(path) and not args.force:
            nskip += 1
            continue
        try:
            state = load_state(sid)
            kw = state["distractor_keyword"]
            with open(S.text_doc_path(sid, k)) as f:
                doc = f.read()
            avoid = re.findall(r"[A-Za-z]+",
                               state["correct_action"] + " " + state["distractor_action"])
            ctrl = S.pick_control_word(doc, kw, k, avoid) if k > 0 else None
            words = [kw] + ([ctrl] if ctrl else [])

            if mod == "text":
                msgs = append_query(S.build_messages_text(state, doc), QUERY)
                ids = S.ids_of(tok, render_readout(tok, msgs))
                res = score_words(model, tok, ids, words, device)
                extra = {"n_prompt_tokens": len(ids)}
            else:
                cards = card_paths_for(sid, k, "native")
                if not cards:
                    with open(path, "w") as f:
                        json.dump({"state_id": sid, "k": k, "modality": mod,
                                   "skipped": "no_cards"}, f)
                    nskip += 1
                    continue
                from PIL import Image
                images = [Image.open(p).convert("RGB") for p in cards]
                msgs = append_query(S.build_messages_visual(state, cards), QUERY)
                # ---- Gemma-4 image protocol --------------------------------
                # The Qwen path used image_grid_thw + merge_size + manual pad
                # expansion.  Gemma-4 has none of those: the processor expands
                # `<|image|>` internally and returns ready-to-use input_ids,
                # plus mm_token_type_ids / image_position_ids.  Using the Qwen
                # keys here raises KeyError: 'image_grid_thw'.
                base = processor(text=[render_readout(processor, msgs)], images=images,
                                 return_tensors="pt")
                ids = base["input_ids"][0].tolist()
                mk = {}
                for kk, vv in base.items():
                    if kk in ("input_ids", "attention_mask") or not hasattr(vv, "to"):
                        continue
                    mk[kk] = vv.to(device=device)
                if "pixel_values" in mk:
                    mk["pixel_values"] = mk["pixel_values"].to(dtype=torch.bfloat16)
                mm = base.get("mm_token_type_ids")
                res = score_words(model, tok, ids, words, device, model_kwargs=mk)
                extra = {"n_prompt_tokens": len(ids), "n_cards": len(cards),
                         "n_image_tokens": (int((mm != 0).sum())
                                            if mm is not None else None),
                         "max_soft_tokens": int(getattr(
                             processor.image_processor, "max_soft_tokens", -1)),
                         "image_backend": "gemma4_processor"}

            rec = {"state_id": sid, "game_name": state.get("game_name"), "k": k,
                   "modality": mod, "keyword": kw, "control_word": ctrl,
                   "readout": res, "query": QUERY, "stem": READOUT_STEM,
                   "logp_keyword": res[kw]["logp"],
                   "n_tokens_keyword": res[kw]["n_tokens"],
                   "logp_control": res[ctrl]["logp"] if ctrl else None,
                   "margin": (res[kw]["logp"] - res[ctrl]["logp"]) if ctrl else None,
                   **extra}
            with open(path, "w") as f:
                json.dump(rec, f)
            ndone += 1
        except Exception:
            nerr += 1
            with open(os.path.join(OUT, "errors.log"), "a") as f:
                f.write(f"=== {sid} k{k} {mod}\n{traceback.format_exc()}\n")
        if (i + 1) % 20 == 0:
            el = time.time() - t0
            print(f"  [{i+1}/{len(cells)}] done={ndone} skip={nskip} err={nerr} "
                  f"{el:.0f}s {el/max(1,ndone):.2f}s/cell", flush=True)

    print(f"[F4/{mod}] FINISHED done={ndone} skip={nskip} err={nerr} "
          f"in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
