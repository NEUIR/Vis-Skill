#!/usr/bin/env python3
"""
symbol-encoding / F2a — render the SAME cased keyword strings into skill cards (CPU).

This is the other half of the crossed design. In text, `cabinet -> CaBiNeT`
multiplies the keyword's BPE tokens (1 -> 4). Rendered to pixels, the identical
string changes segmentation NOT AT ALL — the vision tower sees glyphs and emits a
fixed number of patch tokens per unit area regardless of letter case. So:

    if the text drop is about SEGMENTATION  -> visual is FLAT across casing
    if it is about SURFACE FORM / salience  -> visual drops too

Only that contrast can separate "text rides on native discrete tokens" from
"altcase just makes the word weird".

We reuse `render_cards.render_card` from ../../lexical-interference/code so the
cards are produced by the very same layout engine that made the LI cards (same
fonts, canvas, wrapping). `--verify-native` re-renders the native documents and
byte-compares against the existing LI PNGs; that is the gate proving this script
is layout-identical to the originals, which is what lets the `native` arm reuse
the LI cards instead of re-rendering 825 cells of them.

Output: results/cards/<state_id>/k<k>/<arm>/card<i>_<skill>.png

Run:
  python3 render_cased_cards.py --verify-native --limit 5
  python3 render_cased_cards.py --arms altcase upper --workers 16
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
MECH_ROOT = os.path.dirname(HERE)
sys.path.insert(0, MECH_ROOT)
sys.path.insert(0, HERE)

from lib.config import LI_CODE_DIR, LI_VARIANTS_DIR  # noqa: E402
from lib.cells import load_manifest, load_state  # noqa: E402
import symlib as S  # noqa: E402

sys.path.insert(0, LI_CODE_DIR)
import render_cards as RC  # noqa: E402

CARDS = os.path.join(HERE, "results", "cards")


def parts_path(state_id: str, k: int) -> str:
    return os.path.join(LI_VARIANTS_DIR, "text", state_id, f"level_{k}.parts.json")


def li_card_dir(state_id: str, k: int) -> str:
    return os.path.join(LI_VARIANTS_DIR, "visual", state_id, f"level_{k}")


def out_dir(state_id: str, k: int, arm: str) -> str:
    return os.path.join(CARDS, state_id, f"k{k}", arm)


def recase_parts(parts, keyword: str, arm: str):
    """Recase the keyword in every part's description+markdown.

    Returns (new_parts, n_replaced_total). The invariant checked by the caller is
    n_replaced_total == k: the keyword lives only in the injected sentences, so
    the count across the 5 skill documents must equal the condition's k.
    """
    new, total = [], 0
    for p in parts:
        q = dict(p)
        for field in ("description", "markdown"):
            if q.get(field):
                q[field], n = S.recase_word_in_doc(q[field], keyword, arm)
                total += n
        new.append(q)
    return new, total


def _render_one(job):
    state_id, k, arm, keyword, force = job
    pp = parts_path(state_id, k)
    if not os.path.exists(pp):
        return ("miss", state_id, k, arm, 0)
    with open(pp) as f:
        parts = json.load(f)

    if arm == "native":
        new_parts, n_rep = parts, k          # identity
    else:
        new_parts, n_rep = recase_parts(parts, keyword, arm)
        if n_rep != k:
            return ("badcount", state_id, k, arm, n_rep)

    od = out_dir(state_id, k, arm)
    os.makedirs(od, exist_ok=True)
    n = 0
    for i, part in enumerate(new_parts):
        # filename identical to LI's so downstream [Skill: name] labels match
        op = os.path.join(od, f"card{i}_{part['skill_name']}.png")
        if os.path.exists(op) and not force:
            n += 1
            continue
        RC.render_card(part["skill_name"], part.get("description", ""),
                       part["markdown"], op)
        n += 1
    return ("ok", state_id, k, arm, n)


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=["altcase", "upper"])
    ap.add_argument("--ks", nargs="*", type=int, default=[4, 8])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--verify-native", action="store_true",
                    help="render native and byte-compare with the existing LI cards")
    args = ap.parse_args()

    manifest = load_manifest()
    if args.limit:
        manifest = manifest[:args.limit]
    kw_by_state = {s["state_id"]: load_state(s["state_id"])["distractor_keyword"]
                   for s in manifest}

    if args.verify_native:
        # GATE: our render_card call must reproduce the LI cards exactly.
        jobs = [(s["state_id"], k, "native", kw_by_state[s["state_id"]], True)
                for s in manifest for k in args.ks]
        with Pool(args.workers) as pool:
            list(pool.imap_unordered(_render_one, jobs))
        same = diff = missing = 0
        bad = []
        for sid, k, *_ in [(j[0], j[1]) for j in jobs]:
            od, ld = out_dir(sid, k, "native"), li_card_dir(sid, k)
            if not os.path.isdir(ld):
                missing += 1
                continue
            for fn in sorted(os.listdir(od)):
                lp = os.path.join(ld, fn)
                if not os.path.exists(lp):
                    missing += 1
                    continue
                if sha(os.path.join(od, fn)) == sha(lp):
                    same += 1
                else:
                    diff += 1
                    if len(bad) < 5:
                        bad.append(f"{sid}/k{k}/{fn}")
        print(f"[verify-native] identical={same} different={diff} missing={missing}")
        if bad:
            print("  first mismatches:", bad)
        print("  -> native arm may reuse the LI cards"
              if diff == 0 else "  -> RE-RENDER native too (layout drift!)")
        return

    jobs = [(s["state_id"], k, arm, kw_by_state[s["state_id"]], args.force)
            for s in manifest for k in args.ks for arm in args.arms]
    print(f"[F2a] rendering {len(jobs)} (state,k,arm) card sets on {args.workers} workers",
          flush=True)
    tally = {}
    with Pool(args.workers) as pool:
        for i, (status, sid, k, arm, n) in enumerate(pool.imap_unordered(_render_one, jobs)):
            tally[status] = tally.get(status, 0) + 1
            if status != "ok":
                print(f"  [{status}] {sid} k={k} {arm} n={n}", flush=True)
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(jobs)} {tally}", flush=True)
    print(f"[F2a] done {tally}", flush=True)


if __name__ == "__main__":
    main()
