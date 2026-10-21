#!/usr/bin/env python3
"""
symbol-encoding / F2b — VISUAL casing arm: the crossed half of the decisive test.

Score exactly the same cased documents as F1, but delivered as PIXELS. Casing
changes the text condition's BPE segmentation (1 -> 4 tokens for `cabinet`); it
changes the visual condition's tokenisation not at all, because the vision tower
emits patch tokens per unit AREA, not per symbol.

    arm       text keyword tokens      visual keyword tokens
    native    1.56 (mean)              ~ area / patch size
    upper     2.72                     ~ same
    altcase   4.72                     ~ same

So the prediction that discriminates the hypotheses is an INTERACTION:

    H (text rides on native discrete tokens)
        text slope falls with casing, visual slope FLAT
    surface-form / salience account
        both fall
    symbol crispness irrelevant
        neither falls

All cards are re-rendered by render_cased_cards.py in THIS environment, including
`native`, because a Unicode box-drawing glyph in some code fences resolves to a
different font fallback than when the LI cards were made (cosmetic, left margin,
keyword-independent — but it would make native-vs-altcase not strictly
apples-to-apples if we mixed sources). The substantive gate is therefore the
SLOPE: the native arm on k in {0,4,8} must reproduce LI visual 0.552.

We also record the keyword's visual token footprint per cell so the analysis can
check that casing did not change AREA (uppercase letters are wider — if the
footprint moves, that is a covariate to report, not to hide).

Run:
  MECH_CUDA_INDEX=1 python3 run_visual_casing.py --shard 0 --num-shards 2
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
MECH_ROOT = os.path.dirname(HERE)
sys.path.insert(0, MECH_ROOT)
sys.path.insert(0, HERE)

from lib.cells import load_manifest, load_state  # noqa: E402
import symlib as S  # noqa: E402
# --- Gemma-4 override -------------------------------------------------------
# `symlib.py` in this directory is identical to the Qwen implementation, whose
# score_actions_visual assumes `image_grid_thw` / `merge_size` and dies on
# Gemma-4 with KeyError: 'image_grid_thw'.
# symlib_gemma replaces ONLY the visual scorer; the text path is untouched.
import symlib_gemma  # noqa: E402
symlib_gemma.patch(S)
# ----------------------------------------------------------------------------
from render_cased_cards import out_dir as cards_dir  # noqa: E402

OUT = os.path.join(HERE, "results", "visual_casing")
os.makedirs(OUT, exist_ok=True)

VISUAL_ARMS = ["native", "upper", "altcase"]


def card_paths_for(state_id: str, k: int, arm: str):
    d = cards_dir(state_id, k, "native" if k == 0 else arm)
    return sorted(glob.glob(os.path.join(d, "card*.png")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--states", nargs="*", default=None)
    ap.add_argument("--ks", nargs="*", type=int, default=S.KS)
    ap.add_argument("--arms", nargs="*", default=VISUAL_ARMS)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    manifest = load_manifest()
    sids = [s["state_id"] for s in manifest]
    if args.states:
        sids = [s for s in sids if s in args.states]
    if args.limit:
        sids = sids[:args.limit]
    sids = [s for i, s in enumerate(sids) if i % args.num_shards == args.shard]
    cells = list(S.plan_cells(sids, args.ks, args.arms))
    print(f"[F2b] shard {args.shard}/{args.num_shards} states={len(sids)} cells={len(cells)}",
          flush=True)

    device = f"cuda:{os.environ.get('MECH_CUDA_INDEX', '0')}"
    print(f"[F2b] loading model on {device} ...", flush=True)
    model, processor, tok = S.load_visual_model(device)
    # Hard gate: refuse to burn GPU hours if the Gemma visual override is not
    # installed.
    symlib_gemma.assert_patched(S)
    _mst = symlib_gemma.pin_soft_tokens(processor)
    print(f"[F2b] model ready (max_soft_tokens pinned to {_mst})", flush=True)

    from PIL import Image
    t0 = time.time()
    ndone = nskip = nerr = 0
    for i, (sid, k, arm) in enumerate(cells):
        cid = S.cell_id(sid, k, arm)
        path = os.path.join(OUT, f"{cid}.json")
        if os.path.exists(path) and not args.force:
            nskip += 1
            continue
        try:
            cps = card_paths_for(sid, k, arm)
            if not cps:
                with open(path, "w") as f:
                    json.dump({"state_id": sid, "k": k, "arm": arm,
                               "skipped": "no_cards"}, f)
                nskip += 1
                continue
            state = load_state(sid)
            images = [Image.open(p).convert("RGB") for p in cps]
            messages = S.build_messages_visual(state, cps)
            rec = S.score_actions_visual(model, tok, processor, messages, images,
                                         state["correct_action"],
                                         state["distractor_action"], device)
            rec.update({
                "state_id": sid, "game_name": state.get("game_name"),
                "k": k, "arm": arm, "keyword": state["distractor_keyword"],
                "n_cards": len(cps),
                "card_sizes": [list(im.size) for im in images],
                "correct_action": state["correct_action"],
                "distractor_action": state["distractor_action"],
            })
            with open(path, "w") as f:
                json.dump(rec, f)
            ndone += 1
        except Exception:
            nerr += 1
            with open(os.path.join(OUT, "errors.log"), "a") as f:
                f.write(f"=== {cid}\n{traceback.format_exc()}\n")
        if (i + 1) % 10 == 0:
            el = time.time() - t0
            print(f"  [{i+1}/{len(cells)}] done={ndone} skip={nskip} err={nerr} "
                  f"{el:.0f}s {el/max(1,ndone):.2f}s/cell", flush=True)

    print(f"[F2b] FINISHED done={ndone} skip={nskip} err={nerr} in {time.time()-t0:.0f}s",
          flush=True)


if __name__ == "__main__":
    main()
