"""
Step 2 — build the k = 0/1/2/4/8 skill documents for every decision state.

The manipulation, and nothing but the manipulation:

  base document = the state's top-5 SKILL.md documents (the same five the two
                  main experiments were given), with every whole-word occurrence
                  of the distractor keyword replaced by a single neutral word
                  (`microwave` -> `appliance`, ...). One word for one word, so
                  the base length is untouched and k=0 really is zero mentions.

  N_SLOTS (=8) insertion slots are chosen at fixed, evenly spaced paragraph
  boundaries of that base document. EVERY variant inserts a sentence at ALL
  eight slots. In variant k, k of those sentences are INJECTIONS (each naming
  the distractor action exactly once) and 8-k are the paired FILLERS — same
  slot, same register, same word count, no action mention.

So across k = 0,1,2,4,8 the document has:
  * identical base content,
  * identical number of sentences at identical positions,
  * identical total word count (asserted, not hoped for),
and differs only in how many times `open microwave 1` / `microwave` appears.
That is the clean version of "control the keyword frequency... keep the rest
of the content and the total length as equal as possible".

Which slots carry the injections is chosen by `--slot_order`:
  spread (default) — the k injections are spread evenly over the 8 positions, so
                     k does not covary with *where* in the document the text sits;
  prefix           — the first k slots, i.e. the simpler but position-confounded
                     variant.

Outputs per state:
  variants/text/{state_id}/level_{k}.md          full document (text condition)
  variants/text/{state_id}/level_{k}.parts.json  per-skill split (render input)
  variants/construction_meta.json                measured counts + assertions
"""
import re
import json
import argparse
from pathlib import Path
from collections import Counter

import skill_lexicon as lex


# ══════════════════════════════════════════════════════════════════════
# Markdown block splitting (fence-aware)
# ══════════════════════════════════════════════════════════════════════

def split_blocks(md):
    """Split markdown into blank-line separated blocks, keeping fenced code
    blocks intact."""
    blocks, cur, in_fence = [], [], False
    for line in md.split("\n"):
        if line.strip().startswith("```"):
            in_fence = not in_fence
            cur.append(line)
            continue
        if not line.strip() and not in_fence:
            if cur:
                blocks.append("\n".join(cur).strip("\n"))
                cur = []
            continue
        cur.append(line)
    if cur:
        blocks.append("\n".join(cur).strip("\n"))
    return [b for b in blocks if b.strip()]


def _count_term(text, term):
    """Whole-word count for a single word, phrase count for a multi-word term."""
    return lex.count_phrase(text, term) if " " in term else lex.count_word(text, term)


def is_insertable_after(block):
    """A slot may follow a block of ordinary prose or a finished list, but not a
    bare heading (that would separate a heading from its content) and not a code
    fence (mid-example insertions would read as part of the example)."""
    s = block.strip()
    if s.startswith("#"):
        return False
    if s.startswith("```") or s.endswith("```"):
        return False
    return len(s.split()) >= 8


# ══════════════════════════════════════════════════════════════════════
# Document assembly
# ══════════════════════════════════════════════════════════════════════

def load_parts(corpus, skill_names, keyword):
    """-> list of {skill_name, description, blocks} with the keyword scrubbed."""
    parts = []
    for name in skill_names:
        body = corpus.body(name)
        if not body:
            continue
        desc = corpus.description(name)
        parts.append({
            "skill_name": name,
            "description": lex.scrub_keyword(desc, keyword),
            "blocks": split_blocks(lex.scrub_keyword(body, keyword)),
        })
    return parts


def choose_anchors(parts, n_slots):
    """Pick n_slots evenly spaced insertion anchors across all parts.

    An anchor is (part_index, block_index) = "insert a new paragraph right after
    this block". If a document offers fewer than n_slots usable anchors, the
    remaining slots are appended at the end of the last part; that is recorded in
    the metadata so it never silently changes the design.
    """
    usable = [(pi, bi)
              for pi, p in enumerate(parts)
              for bi, b in enumerate(p["blocks"]) if is_insertable_after(b)]
    if not usable:
        end = [(len(parts) - 1, len(parts[-1]["blocks"]) - 1)] if parts else [(0, 0)]
        return end * n_slots, n_slots

    if len(usable) >= n_slots:
        step = len(usable) / n_slots
        chosen = [usable[min(len(usable) - 1, int((i + 0.5) * step))] for i in range(n_slots)]
        # keep them distinct where possible
        out, used = [], set()
        for a in chosen:
            while a in used:
                pi, bi = a
                nxt = [u for u in usable if u not in used and u >= (pi, bi)]
                a = nxt[0] if nxt else next(u for u in usable if u not in used)
            used.add(a)
            out.append(a)
        return sorted(out), 0

    pad = n_slots - len(usable)
    return sorted(usable) + [usable[-1]] * pad, pad


def build_document(parts, anchors, injected_slots, pairs):
    """Assemble the full document and its per-part split.

    `anchors[i]` is where slot i goes; slot i uses pairs[i][0] (injection) when
    i is in `injected_slots`, else pairs[i][1] (filler).
    """
    after = {}
    for slot, anchor in enumerate(anchors):
        text = pairs[slot][0] if slot in injected_slots else pairs[slot][1]
        after.setdefault(anchor, []).append(text)

    out_parts = []
    for pi, p in enumerate(parts):
        lines = []
        for bi, block in enumerate(p["blocks"]):
            lines.append(block)
            for extra in after.get((pi, bi), []):
                lines.append(extra)
        out_parts.append({
            "skill_name": p["skill_name"],
            "description": p["description"],
            "markdown": "\n\n".join(lines).strip() + "\n",
        })

    doc = ["# Relevant Skills for This Task\n"]
    for p in out_parts:
        doc.append(f"=== Skill: {p['skill_name']} ===")
        if p["description"]:
            doc.append(f"Description: {p['description']}\n")
        doc.append(p["markdown"])
    return "\n".join(doc).strip() + "\n", out_parts


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main(args):
    states_dir = Path(args.states_dir)
    out_dir = Path(args.output_dir)
    text_dir = out_dir / "text"
    text_dir.mkdir(parents=True, exist_ok=True)

    levels = [int(x) for x in args.levels.split(",")]
    if max(levels) > lex.N_SLOTS:
        raise SystemExit(f"levels must not exceed N_SLOTS={lex.N_SLOTS}; got {levels}")

    manifest = json.loads((states_dir / "manifest.json").read_text(encoding="utf-8"))
    corpus = lex.SkillCorpus(args.skills_dir)
    print(f"[INFO] {manifest['total_states']} states, levels={levels}, "
          f"slot_order={args.slot_order}, skills corpus={len(corpus)}")

    all_meta = []
    problems = Counter()

    for info in manifest["states"]:
        state_id = info["state_id"]
        state_file = states_dir / f"{state_id}.json"
        if not state_file.exists():
            problems["missing_state_file"] += 1
            continue
        state = json.loads(state_file.read_text(encoding="utf-8"))

        keyword = state["distractor_keyword"]
        phrase = state["distractor_phrase"]
        action = state["distractor_action"]
        skill_names = state.get("relevant_skills") or []

        parts = load_parts(corpus, skill_names, keyword)
        if not parts:
            problems["no_skill_bodies"] += 1
            continue

        anchors, padded_slots = choose_anchors(parts, lex.N_SLOTS)
        pairs = lex.slot_texts(action, keyword=keyword)

        # The scrub only rewrites bodies and descriptions; a skill NAME carrying
        # the keyword would leak k+1 mentions into the document header and the
        # card title. No name in this corpus does, but assert rather than assume.
        for p in parts:
            if lex.count_word(p["skill_name"], keyword):
                raise SystemExit(
                    f"[ERROR] {state_id}: skill name '{p['skill_name']}' contains the "
                    f"distractor keyword '{keyword}' — it is not scrubbed, so the "
                    f"k = 0 document would not have zero mentions.")

        state_out = text_dir / state_id
        state_out.mkdir(parents=True, exist_ok=True)

        per_level = {}
        for k in levels:
            injected = set(lex.slot_assignment(k, lex.N_SLOTS, mode=args.slot_order))
            doc, out_parts = build_document(parts, anchors, injected, pairs)
            (state_out / f"level_{k}.md").write_text(doc, encoding="utf-8")
            (state_out / f"level_{k}.parts.json").write_text(
                json.dumps(out_parts, indent=2, ensure_ascii=False), encoding="utf-8")

            per_level[k] = {
                "injected_slots": sorted(injected),
                "keyword_count": lex.count_word(doc, keyword),
                "phrase_count": lex.count_phrase(doc, phrase),
                "action_count": doc.count(action),
                "verb_count": _count_term(doc, lex.VERB_SURFACE.get(
                    state["distractor_action_type"], state["distractor_action_type"])),
                "words": lex.word_count(doc),
                "chars": len(doc),
                "n_parts": len(out_parts),
            }

        # ── verification: the manipulation is exactly what it claims to be ──
        checks = {}
        checks["keyword_count_equals_k"] = all(
            per_level[k]["keyword_count"] == k for k in levels)
        # The literal distractor action string appears exactly k times. Together
        # with keyword_count this is the manipulation, stated two ways.
        checks["action_count_equals_k"] = all(
            per_level[k]["action_count"] == k for k in levels)
        checks["phrase_count_equals_k"] = all(
            per_level[k]["phrase_count"] == k for k in levels)
        # "{verb} {noun}" is only contiguous in the ONE-argument grammar
        # ("open microwave 1"). The two-argument grammar interleaves the object
        # ("clean candle 1 with sinkbasin 2"), so phrase_count is legitimately 0
        # there — informational, never a gate. See lex.count_verb_noun.
        checks["phrase_count_is_gate"] = " with " not in action
        word_counts = {per_level[k]["words"] for k in levels}
        checks["word_count_constant"] = len(word_counts) == 1
        char_counts = [per_level[k]["chars"] for k in levels]
        checks["max_char_delta"] = max(char_counts) - min(char_counts)
        checks["max_char_delta_pct"] = round(
            100.0 * checks["max_char_delta"] / max(1, min(char_counts)), 3)
        for name in ("keyword_count_equals_k", "action_count_equals_k",
                     "word_count_constant"):
            if not checks[name]:
                problems[name] += 1
        if checks["phrase_count_is_gate"] and not checks["phrase_count_equals_k"]:
            problems["phrase_count_equals_k"] += 1

        all_meta.append({
            "state_id": state_id,
            "distractor_action": action,
            "distractor_keyword": keyword,
            "distractor_phrase": phrase,
            "skills_used": [p["skill_name"] for p in parts],
            "anchors": [list(a) for a in anchors],
            "padded_slots": padded_slots,
            "levels": {str(k): per_level[k] for k in levels},
            "checks": checks,
        })
        if padded_slots:
            problems["padded_slots"] += 1

    meta = {
        "total_states": len(all_meta),
        "levels": levels,
        "n_slots": lex.N_SLOTS,
        "slot_order": args.slot_order,
        "skills_dir": args.skills_dir,
        "problems": dict(problems),
        "states": all_meta,
    }
    (out_dir / "construction_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 68)
    print(f"Constructed {len(all_meta)} × {len(levels)} documents -> {text_dir}")
    if all_meta:
        for k in levels:
            ws = [m["levels"][str(k)]["words"] for m in all_meta]
            cs = [m["levels"][str(k)]["chars"] for m in all_meta]
            kc = [m["levels"][str(k)]["keyword_count"] for m in all_meta]
            print(f"  k={k}: words {sum(ws)/len(ws):8.0f} | chars {sum(cs)/len(cs):8.0f} "
                  f"| keyword count {min(kc)}..{max(kc)} (must be {k})")
        deltas = [m["checks"]["max_char_delta_pct"] for m in all_meta]
        print(f"  within-state length spread across k: "
              f"max {max(deltas):.2f}% (mean {sum(deltas)/len(deltas):.2f}%)")
    print(f"  problems: {dict(problems) or 'none'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--states_dir", required=True)
    p.add_argument("--skills_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--levels", default="0,1,2,4,8")
    p.add_argument("--slot_order", choices=["spread", "prefix"], default="spread")
    main(p.parse_args())
