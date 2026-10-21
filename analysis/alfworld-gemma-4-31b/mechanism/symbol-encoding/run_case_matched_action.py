#!/usr/bin/env python3
"""
symbol-encoding / F3 — CASE-MATCHED ACTION: is the text carrier symbol IDENTITY or symbol
COUNT?

F1's interim result forced this experiment. In F1 the `title` arm costs almost as
much slope as `upper`/`altcase`, yet `title` leaves the keyword at ONE token for
most keywords (cabinet -> Cabinet, safe -> Safe). And the F1 regression of slope
on the MEASURED keyword token count is flat/non-monotone. So the ~20% cost is
not fragmentation-per-se. The remaining candidate is token IDENTITY: in F1 every
non-native arm breaks the exact match between the token in the DOCUMENT and the
token the model must EMIT in the action (which stayed lowercase by design).

F3 tests that directly by restoring the match instead of breaking it further:

  arm            document               scored distractor action     identity match?
  native   (F1)  ... open cabinet 4 ... "open cabinet 4"             YES
  upper    (F1)  ... open CABINET 4 ... "open cabinet 4"             no
  match_upper    ... open CABINET 4 ... "open CABINET 4"             YES (restored)
  altcase  (F1)  ... open CaBiNeT 4 ... "open cabinet 4"             no
  match_altcase  ... open CaBiNeT 4 ... "open CaBiNeT 4"             YES (restored)

Predictions
  IDENTITY  is the carrier: slope(match_*) ~ slope(native)  >>  slope(upper/altcase).
            Re-pairing the symbol pays the cost back even though the surface is
            still mangled and (for altcase) still 4-6 tokens.
  COUNT/CRISPNESS is the carrier: slope(match_altcase) stays LOW — a 6-token
            mangled symbol cannot carry exposure no matter what we score, and
            match_upper lands in between.
  Neither:  match arms also ~native and F1's drop was something else entirely
            (then trust ctrl_altcase = -2.4% as the scale of document-level noise).

METHOD NOTE — each match arm gets its OWN k=0 anchor. Recasing the scored action
changes the action's token sequence and therefore its absolute logP and its token
count, so a match arm may NOT share the native k=0 cell. Slopes are differences
against a same-tokenization anchor, which is what makes them comparable across
arms; absolute logP across arms is not comparable and is not compared.

The correct action is recased with the SAME rule (whole-word, case-insensitive)
so that if it happens to contain the keyword it is treated identically; logP
(correct) remains a control that must stay ~0-slope.

Run:
  MECH_CUDA_INDEX=0 python3 run_case_matched_action.py --shard 0 --num-shards 2
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
from run_text_casing import build_doc  # reuse the exact document builder + invariants

OUT = os.path.join(HERE, "results", "case_matched")
os.makedirs(OUT, exist_ok=True)

# arm -> (document keyword casing, scored action casing)
#
# The match_* arms alone are NOT interpretable: recasing the scored action changes
# its token count AND its prior, so its k=0 baseline is far lower and its dynamic
# range far larger than native's. Comparing slopes across different scored strings
# therefore inflates any "rescue" (measured: 7-16x, not ~1).
#
# The mismatch_* arms fix that. They score the SAME recased action but leave the
# document NATIVE lowercase. So match_X vs mismatch_X holds the scored string, its
# token count, its prior, k, and the k=0 anchor all identical, and varies ONLY
# whether the document contains the identical symbol. That difference IS the
# identity effect, with no dynamic-range confound.
ARM_SPEC = {
    "match_upper":      ("upper",  "upper"),
    "match_altcase":    ("altcase", "altcase"),
    "mismatch_upper":   ("native", "upper"),
    "mismatch_altcase": ("native", "altcase"),
}
MATCH_ARMS = {k: v[1] for k, v in ARM_SPEC.items()}   # back-compat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--states", nargs="*", default=None)
    ap.add_argument("--ks", nargs="*", type=int, default=S.KS)
    ap.add_argument("--arms", nargs="*", default=sorted(ARM_SPEC))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    manifest = load_manifest()
    sids = [s["state_id"] for s in manifest]
    if args.states:
        sids = [s for s in sids if s in args.states]
    if args.limit:
        sids = sids[:args.limit]
    sids = [s for i, s in enumerate(sids) if i % args.num_shards == args.shard]

    # NOT S.plan_cells: every arm needs its OWN k=0 anchor here (see METHOD NOTE).
    cells = [(sid, k, arm) for sid in sids for arm in args.arms for k in args.ks]
    print(f"[F3] shard {args.shard}/{args.num_shards} states={len(sids)} cells={len(cells)}",
          flush=True)

    if args.dry_run:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(S.MODEL_PATH, trust_remote_code=True)
        seen, skips = {}, {}
        for sid, k, arm in cells:
            st = load_state(sid)
            doc, meta = build_doc(st, k, ARM_SPEC[arm][0], tok)
            if doc is None:
                skips[meta] = skips.get(meta, 0) + 1
                continue
            d_act, n_act = S.recase_word_in_doc(st["distractor_action"],
                                                st["distractor_keyword"], ARM_SPEC[arm][1])
            key = (arm, k)
            if key not in seen:
                seen[key] = (d_act, n_act, meta.get("kw_token_count"))
        for key in sorted(seen):
            print(f"  {key}  action={seen[key][0]!r} n_rep={seen[key][1]} "
                  f"kw_tokens={seen[key][2]}")
        print(f"  skips: {skips}  cells={len(cells)}")
        return

    device = f"cuda:{os.environ.get('MECH_CUDA_INDEX', '0')}"
    print(f"[F3] loading model on {device} ...", flush=True)
    model, tok = S.load_text_model(device)
    print("[F3] model ready", flush=True)

    t0 = time.time()
    ndone = nskip = nerr = 0
    for i, (sid, k, arm) in enumerate(cells):
        cid = S.cell_id(sid, k, arm)
        path = os.path.join(OUT, f"{cid}.json")
        if os.path.exists(path) and not args.force:
            nskip += 1
            continue
        doc_casing, casing = ARM_SPEC[arm]
        try:
            state = load_state(sid)
            doc, meta = build_doc(state, k, doc_casing, tok)
            if doc is None:
                with open(path, "w") as f:
                    json.dump({"state_id": sid, "k": k, "arm": arm, "skipped": meta}, f)
                nskip += 1
                continue
            kw = state["distractor_keyword"]
            # Recase the keyword inside the SCORED actions with the same rule, so
            # the emitted symbol is identical to the one in the document.
            d_act, n_d = S.recase_word_in_doc(state["distractor_action"], kw, casing)
            c_act, n_c = S.recase_word_in_doc(state["correct_action"], kw, casing)
            meta.update({
                "casing": casing,
                "doc_casing": doc_casing,
                "identity_match": doc_casing == casing,
                "distractor_action_scored": d_act,
                "correct_action_scored": c_act,
                "n_recased_in_distractor_action": n_d,
                "n_recased_in_correct_action": n_c,
                # the keyword must actually occur in the distractor action, else
                # this arm is a no-op and the cell is not informative
                "identity_restored": n_d > 0,
            })
            messages = S.build_messages_text(state, doc)
            rec = S.score_actions_text(model, tok, messages, c_act, d_act, device)
            rec.update({"state_id": sid, "game_name": state.get("game_name"),
                        "k": k, "arm": arm, "meta": meta,
                        "correct_action": c_act, "distractor_action": d_act,
                        "correct_action_native": state["correct_action"],
                        "distractor_action_native": state["distractor_action"]})
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

    print(f"[F3] FINISHED done={ndone} skip={nskip} err={nerr} in {time.time()-t0:.0f}s",
          flush=True)


if __name__ == "__main__":
    main()
