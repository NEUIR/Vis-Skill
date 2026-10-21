#!/usr/bin/env python3
"""
symbol-encoding / F1 — TEXT casing ladder: does the text exposure effect need the keyword
to be a NATIVE DISCRETE TOKEN?

For each (state, k, arm) we rewrite the k injected keyword occurrences under a
casing that changes ONLY the BPE segmentation, then score
logP(distractor action) and logP(correct action) with the stem-only
forced-decoding span.

    arm            document                       keyword tokens (e.g. cabinet)
    native         ... open cabinet 4 ...         1
    title          ... open Cabinet 4 ...         1     <- surface changed, still 1 token
    upper          ... open CABINET 4 ...         2
    altcase        ... open CaBiNeT 4 ...         4
    ctrl_altcase   keyword native, a FREQUENCY-MATCHED other word mangled

Held constant across arms: characters, word, semantics, human legibility, word
count, slot positions, k, and the SCORED ACTION STRING (always the original
lowercase "open cabinet 4"). Only segmentation moves.

Predictions
  H (native discrete symbol): slope falls as the measured keyword token count
      rises. native ~ title > upper > altcase. ctrl_altcase ~ native.
  Not-H (surface mismatch):   title falls as much as upper/altcase.
  Not-H (nothing to do with
         symbol crispness):   all arms flat, and the k=0 anchor plus the native
      slope still reproduce LI (this is also the built-in GATE).

GATE: the `native` arm on k in {0,4,8} must reproduce the stem-only text
baseline 0.846 (LI same-subset 0.847). If it does not, nothing else here is
interpretable and the run should be stopped.

Run:
  MECH_CUDA_INDEX=0 python3 run_text_casing.py --shard 0 --num-shards 2
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
MECH_ROOT = os.path.dirname(HERE)
sys.path.insert(0, MECH_ROOT)
sys.path.insert(0, HERE)

from lib.cells import load_manifest, load_state  # noqa: E402
import symlib as S  # noqa: E402

OUT = os.path.join(HERE, "results", "text_casing")
os.makedirs(OUT, exist_ok=True)


def build_doc(state, k: int, arm: str, tok):
    """-> (doc, meta) or (None, reason). Pure CPU; also does the invariant checks."""
    path = S.text_doc_path(state["state_id"], k)
    if not os.path.exists(path):
        return None, "no_variant"
    with open(path) as f:
        doc = f.read()

    kw = state["distractor_keyword"]
    n_kw = len(re.findall(rf"\b{re.escape(kw)}\b", doc, flags=re.I))
    meta = {"keyword": kw, "n_keyword_occurrences": n_kw,
            "keyword_count_equals_k": (n_kw == k)}

    if k == 0:
        # No keyword to recase; this cell is the shared anchor.
        meta["kw_rendered"] = None
        meta["kw_token_count"] = 0
        meta["control_word"] = None
        return doc, meta

    if n_kw != k:
        # The whole design rests on "keyword count == k". Record and skip rather
        # than silently scoring a mismatched cell.
        return None, f"keyword_count_{n_kw}_ne_k_{k}"

    avoid = re.findall(r"[A-Za-z]+", state["correct_action"] + " " + state["distractor_action"])

    if arm == "ctrl_altcase":
        ctrl = S.pick_control_word(doc, kw, k, avoid)
        if ctrl is None:
            return None, "no_freq_matched_control_word"
        new_doc, n_rep = S.recase_word_in_doc(doc, ctrl, "altcase")
        if n_rep != k:
            return None, f"ctrl_replaced_{n_rep}_ne_k_{k}"
        meta.update({
            "control_word": ctrl,
            "control_rendered": S.CASINGS["altcase"](ctrl),
            "control_token_count_native": S.keyword_token_count(tok, ctrl),
            "control_token_count_mangled": S.keyword_token_count(tok, S.CASINGS["altcase"](ctrl)),
            "kw_rendered": kw,
            "kw_token_count": S.keyword_token_count(tok, kw),
            "n_replaced": n_rep,
        })
        return new_doc, meta

    new_doc, n_rep = S.recase_word_in_doc(doc, kw, arm)
    if n_rep != k:
        return None, f"kw_replaced_{n_rep}_ne_k_{k}"
    rendered = S.CASINGS[arm](kw)
    meta.update({
        "control_word": None,
        "kw_rendered": rendered,
        "kw_token_count": S.keyword_token_count(tok, rendered),
        "kw_token_count_native": S.keyword_token_count(tok, kw),
        "n_replaced": n_rep,
        # sanity: recasing must not change the character count of the document
        "doc_len_unchanged": len(new_doc) == len(doc),
    })
    return new_doc, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--states", nargs="*", default=None)
    ap.add_argument("--ks", nargs="*", type=int, default=S.KS)
    ap.add_argument("--arms", nargs="*", default=S.ARMS)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="build every document on CPU and report the invariants; no model")
    args = ap.parse_args()

    manifest = load_manifest()
    sids = [s["state_id"] for s in manifest]
    if args.states:
        sids = [s for s in sids if s in args.states]
    if args.limit:
        sids = sids[:args.limit]
    # state-major sharding keeps each shard's states COMPLETE across arms, so a
    # partial run is still a balanced panel
    sids = [s for i, s in enumerate(sids) if i % args.num_shards == args.shard]
    cells = list(S.plan_cells(sids, args.ks, args.arms))
    print(f"[F1] shard {args.shard}/{args.num_shards}  states={len(sids)}  cells={len(cells)}",
          flush=True)

    if args.dry_run:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(S.MODEL_PATH, trust_remote_code=True)
        stats, skips = {}, {}
        for sid, k, arm in cells:
            doc, meta = build_doc(load_state(sid), k, arm, tok)
            if doc is None:
                skips[meta] = skips.get(meta, 0) + 1
                continue
            stats.setdefault(arm, []).append(meta.get("kw_token_count", 0))
        for arm, v in sorted(stats.items()):
            import statistics
            print(f"  {arm:14s} n={len(v):4d}  kw_tokens mean={statistics.mean(v):.2f} "
                  f"min={min(v)} max={max(v)}")
        print(f"  skips: {skips}")
        return

    device = f"cuda:{os.environ.get('MECH_CUDA_INDEX', '0')}"
    print(f"[F1] loading model on {device} ...", flush=True)
    model, tok = S.load_text_model(device)
    print("[F1] model ready", flush=True)

    t0 = time.time()
    ndone = nskip = nerr = 0
    for i, (sid, k, arm) in enumerate(cells):
        cid = S.cell_id(sid, k, arm)
        path = os.path.join(OUT, f"{cid}.json")
        if os.path.exists(path) and not args.force:
            nskip += 1
            continue
        try:
            state = load_state(sid)
            doc, meta = build_doc(state, k, arm, tok)
            if doc is None:
                with open(path, "w") as f:
                    json.dump({"state_id": sid, "k": k, "arm": arm,
                               "skipped": meta}, f)
                nskip += 1
                continue
            messages = S.build_messages_text(state, doc)
            rec = S.score_actions_text(model, tok, messages,
                                       state["correct_action"],
                                       state["distractor_action"], device)
            rec.update({"state_id": sid, "game_name": state.get("game_name"),
                        "k": k, "arm": arm, "meta": meta,
                        "correct_action": state["correct_action"],
                        "distractor_action": state["distractor_action"]})
            with open(path, "w") as f:
                json.dump(rec, f)
            ndone += 1
        except Exception:
            nerr += 1
            with open(os.path.join(OUT, "errors.log"), "a") as f:
                f.write(f"=== {cid}\n{traceback.format_exc()}\n")
        if (i + 1) % 25 == 0:
            el = time.time() - t0
            print(f"  [{i+1}/{len(cells)}] done={ndone} skip={nskip} err={nerr} "
                  f"{el:.0f}s  {el/max(1,ndone):.2f}s/cell", flush=True)

    print(f"[F1] FINISHED done={ndone} skip={nskip} err={nerr} in {time.time()-t0:.0f}s",
          flush=True)


if __name__ == "__main__":
    main()
