"""
Smoke test — verify the whole rasterized-interference measurement chain on ONE
state before committing to the full probe run.

Offline checks (no server needed):
  1. states + manifest exist, one state can be loaded; the distractor is NOT
     admissible in that state (the extraction constraint still holds)
  2. text variants exist for every requested k and
       keyword count == k, literal action count == k, word count constant
  3. rasterized cards exist for every k, one per skill part, the same number as
     the structural cards; the render is LOSSLESS (drawn non-whitespace chars
     == source non-whitespace chars) — checked from render_verification.json
     and re-verified in memory on one live re-render
  4. structural cards exist (from the lexical-interference variants)

Online checks (needs a vLLM server on --port):
  5. TEXT / RASTERIZED / STRUCTURAL, k=0 vs k=kmax: forced-decoding
     logP(distractor) / logP(correct) usable, stem is a token prefix,
     /tokenize and prompt_logprobs agree
  6. full candidate-set distribution normalises (rasterized, k=0)
  7. free generation parses into a valid action (rasterized, k=kmax)

Soft failures print WARN (a genuine null result is possible); hard failures
print FAIL and set the exit code.

Usage:
    python smoke_test.py --states_dir ../states --variants_dir ../variants \
        [--state_id ...] [--port 8000] [--skip_online] [--levels 0,1,2,4,8]
"""
import os
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import skill_lexicon as lex  # noqa: E402

N_PASS = N_WARN = N_FAIL = 0


def report(status, msg):
    global N_PASS, N_WARN, N_FAIL
    if status == "PASS":
        N_PASS += 1
    elif status == "WARN":
        N_WARN += 1
    else:
        N_FAIL += 1
    print(f"  [{status}] {msg}")


# ══════════════════════════════════════════════════════════════════════
# Offline checks
# ══════════════════════════════════════════════════════════════════════

def pick_state(states_dir, state_id):
    states_dir = Path(states_dir)
    manifest = states_dir / "manifest.json"
    if not manifest.exists():
        report("FAIL", f"no manifest at {manifest}")
        return None
    entries = json.loads(manifest.read_text(encoding="utf-8"))["states"]
    if not entries:
        report("FAIL", "manifest lists 0 states")
        return None
    if state_id:
        entry = next((e for e in entries if e["state_id"] == state_id), None)
        if entry is None:
            report("FAIL", f"state_id {state_id} not in manifest")
            return None
    else:
        entry = next((e for e in entries
                      if e.get("distractor_keyword") in ("microwave", "fridge")),
                     entries[0])
    sf = states_dir / f"{entry['state_id']}.json"
    if not sf.exists():
        report("FAIL", f"state file missing: {sf}")
        return None
    state = json.loads(sf.read_text(encoding="utf-8"))
    report("PASS", f"state {state['state_id']}: task={state['task_type']} "
                   f"correct='{state['correct_action']}' "
                   f"distractor='{state['distractor_action']}' "
                   f"keyword='{state['distractor_keyword']}' "
                   f"candidates={state['n_candidates']}")
    if state["correct_action"] in state["admissible_commands"]:
        report("PASS", "correct action is in the admissible set")
    else:
        report("FAIL", "correct action NOT admissible — distribution would be ill-defined")
    if state["distractor_action"] not in state["admissible_commands"]:
        report("PASS", "distractor action is NOT admissible now (as designed)")
    else:
        report("FAIL", "distractor is admissible in this state — extraction constraint broken")
    return state


def check_variants(variants_dir, state, levels):
    sid = state["state_id"]
    keyword = state["distractor_keyword"]
    action = state["distractor_action"]
    text_dir = Path(variants_dir) / "text" / sid
    struct_dir = Path(variants_dir) / "visual" / sid
    raster_dir = Path(variants_dir) / "rasterized" / sid

    docs, ok = {}, True
    word_counts = set()
    for k in levels:
        md = text_dir / f"level_{k}.md"
        if not md.exists():
            report("FAIL", f"missing text variant {md}")
            ok = False
            continue
        doc = md.read_text(encoding="utf-8")
        docs[k] = doc
        kc = lex.count_word(doc, keyword)
        ac = doc.count(action)
        word_counts.add(lex.word_count(doc))
        if kc != k:
            report("FAIL", f"k={k}: keyword '{keyword}' count is {kc}, expected {k}")
            ok = False
        if ac != k:
            report("FAIL", f"k={k}: literal action '{action}' count is {ac}, expected {k}")
            ok = False
    if ok:
        report("PASS", f"keyword/action counts == k for all levels {levels}")
    if len(word_counts) <= 1:
        report("PASS", f"word count constant across k ({word_counts})")
    else:
        report("FAIL", f"word count varies across k: {sorted(word_counts)} — length confound!")

    n_struct, n_raster = {}, {}
    for k in levels:
        s_cards = sorted((struct_dir / f"level_{k}").glob("card*.png")) if struct_dir.exists() else []
        r_cards = sorted((raster_dir / f"level_{k}").glob("card*.png")) if raster_dir.exists() else []
        n_struct[k], n_raster[k] = len(s_cards), len(r_cards)
        if not r_cards:
            report("FAIL", f"no rasterized cards at {raster_dir / f'level_{k}'}")
        if not s_cards:
            report("FAIL", f"no structural cards at {struct_dir / f'level_{k}'} "
                           f"(link/keep the lexical-interference visual variants)")
    if all(v > 0 for v in n_raster.values()) and len(set(n_raster.values())) == 1:
        report("PASS", f"rasterized cards present and consistent: "
                       f"{n_raster[levels[0]]} per level")
    if n_struct and n_raster and all(n_struct[k] == n_raster[k] for k in levels
                                     if n_struct.get(k) and n_raster.get(k)):
        report("PASS", "rasterized and structural card counts match per level "
                       "(same images-per-cell budget)")
    elif n_struct:
        report("WARN", f"card counts differ between conditions: struct {n_struct} "
                       f"vs raster {n_raster}")

    # Lossless-render check: the verification file written by render_rasterized.py
    vfile = Path(variants_dir) / "rasterized" / "render_verification.json"
    if vfile.exists():
        v = json.loads(vfile.read_text(encoding="utf-8"))
        if v.get("mismatches", 1) == 0:
            report("PASS", f"render verification: {v['cards']} cards, 0 character "
                           f"mismatches (rasterization is lossless)")
        else:
            report("FAIL", f"render verification reports {v['mismatches']} cards "
                           f"that do NOT preserve the source characters")
    else:
        report("WARN", f"{vfile} missing — rerun render_rasterized.py with --verify")

    # Live re-render of one card in memory: drawn chars must equal source chars.
    parts_file = text_dir / f"level_{levels[-1]}.parts.json"
    if parts_file.exists():
        import render_rasterized as rr
        from PIL import Image, ImageDraw
        parts = json.loads(parts_file.read_text(encoding="utf-8"))
        p0 = parts[0]
        img = Image.new("RGB", (rr.CANVAS_W, 64), rr.BG)
        draw = ImageDraw.Draw(img)
        lines = rr.card_source_lines(p0["skill_name"], p0.get("description", ""),
                                     p0["markdown"])
        drawn = sum(rr._nonspace(r)
                    for ln in lines
                    for r in rr.wrap_line(draw, ln, rr.FONT_MONO,
                                          rr.CANVAS_W - 2 * rr.MARGIN_X))
        source = sum(rr._nonspace(ln) for ln in lines)
        if drawn == source:
            report("PASS", f"live re-render lossless: {drawn} non-whitespace chars "
                           f"drawn == source")
        else:
            report("FAIL", f"live re-render lost characters: drawn {drawn} vs source {source}")
    return docs


# ══════════════════════════════════════════════════════════════════════
# Online checks
# ══════════════════════════════════════════════════════════════════════

def check_online(state, docs, variants_dir, levels, model, port,
                 template="thought_action"):
    import requests
    import probe_lib as pl

    try:
        r = requests.get(f"http://localhost:{port}/health", timeout=5)
        if r.status_code != 200:
            report("FAIL", f"vLLM server on port {port} not healthy — run start_vllm.sh")
            return
    except Exception as e:
        report("FAIL", f"vLLM server on port {port} not reachable ({e})")
        return
    pl.set_endpoint(port)

    sid = state["state_id"]
    correct = lex.normalize_action(state["correct_action"])
    distractor = lex.normalize_action(state["distractor_action"])
    candidates = list(state["candidate_actions"])
    k0, kmax = levels[0], levels[-1]

    for rep in ("text", "rasterized", "structural"):
        print(f"\n  --- {rep.upper()} condition ---")
        results = {}
        for k in (k0, kmax):
            if rep == "text":
                if not docs.get(k):
                    report("FAIL", f"text k={k}: document missing, cannot probe")
                    continue
                messages = pl.build_messages(state, "text", skill_text=docs.get(k))
            else:
                subdir = "rasterized" if rep == "rasterized" else "visual"
                card_dir = Path(variants_dir) / subdir / sid / f"level_{k}"
                cards = sorted(card_dir.glob("card*.png"))
                if not cards:
                    report("FAIL", f"{rep} k={k}: no cards to send ({card_dir})")
                    continue
                messages = pl.build_messages(state, rep,
                                             card_paths=[str(c) for c in cards])

            two, _ = pl.score_candidates(messages, [distractor, correct],
                                         model=model, template=template)
            lp_d, lp_c = two[distractor], two[correct]
            if lp_d.get("stem_prefix_ok") is False or lp_c.get("stem_prefix_ok") is False:
                report("FAIL", f"{rep} k={k}: the forced stem is not a token prefix of "
                               f"the forced turn — logP(action) cannot be isolated")
                continue
            if lp_d.get("tokenization_length_ok") is False:
                report("FAIL", f"{rep} k={k}: /tokenize and prompt_logprobs disagree on "
                               f"the token count — scores would be silently truncated")
                continue
            if not lp_d["complete"] or lp_d["action_logprob"] is None:
                report("FAIL", f"{rep} k={k}: forced decoding of distractor incomplete "
                               f"(matched {lp_d['matched_tokens']}/{lp_d['n_tokens']})")
                continue
            if not lp_c["complete"] or lp_c["action_logprob"] is None:
                report("FAIL", f"{rep} k={k}: forced decoding of correct incomplete")
                continue
            results[k] = {"dist": lp_d["action_logprob"], "corr": lp_c["action_logprob"]}
            print(f"    k={k}: logP(distractor)={lp_d['action_logprob']:.4f}  "
                  f"logP(correct)={lp_c['action_logprob']:.4f}  "
                  f"(n_action_tokens={lp_d['n_action_tokens']})")

        if len(results) == 2:
            report("PASS", f"{rep}: forced decoding works at k={k0} and k={kmax}")
            dd = results[kmax]["dist"] - results[k0]["dist"]
            if rep == "text" and dd <= 0:
                report("WARN", f"text logP(distractor) did NOT increase from k={k0} "
                               f"to k={kmax} (Δ={dd:+.4f}) — possible null result")
            else:
                report("PASS", f"{rep}: ΔlogP(distractor) k{k0}→k{kmax} = {dd:+.4f}")
        elif results:
            report("WARN", f"{rep}: only {len(results)}/2 levels usable")

        # Full candidate-set distribution at k=0 (normalisation sanity), for the
        # NEW condition specifically — text/structural were validated by the
        # lexical-interference run.
        if rep == "rasterized":
            card_dir = Path(variants_dir) / "rasterized" / sid / f"level_{k0}"
            cards = sorted(card_dir.glob("card*.png"))
            if cards:
                messages = pl.build_messages(state, rep, card_paths=[str(c) for c in cards])
                scores, _ = pl.score_candidates(messages, candidates, model=model,
                                                template=template)
                stats = pl.distribution_stats(scores, correct, distractor)
                if stats.get("n_scored") == len(candidates):
                    psum = sum(stats["p_sum"].values())
                    report("PASS", f"rasterized: full candidate set scored "
                                   f"({stats['n_scored']}/{len(candidates)}), "
                                   f"Σp={psum:.6f}, H={stats['entropy_sum']:.3f} nats, "
                                   f"argmax='{stats['argmax_sum']}'")
                    if abs(psum - 1.0) > 1e-6:
                        report("FAIL", "normalised distribution does not sum to 1")
                else:
                    report("FAIL", f"rasterized: only {stats.get('n_scored')}/{len(candidates)} "
                                   f"candidates scored; incomplete: "
                                   f"{stats.get('incomplete_candidates')}")

    # Free generation parse check (rasterized, k=kmax)
    card_dir = Path(variants_dir) / "rasterized" / sid / f"level_{kmax}"
    cards = sorted(card_dir.glob("card*.png"))
    if cards:
        messages = pl.build_messages(state, "rasterized",
                                     card_paths=[str(c) for c in cards])
        text, _ = pl.generate_action(messages, model=model, max_tokens=256)
        gen = pl.summarise_generation(text, correct, distractor, candidates)
        if gen["chosen_action"]:
            report("PASS", f"free generation parses (rasterized k={kmax}): "
                           f"'{gen['chosen_action']}' (correct={gen['chose_correct']}, "
                           f"distractor={gen['chose_distractor']})")
        else:
            report("WARN", f"free generation did not parse an action; raw: {text[:120]!r}")


# ══════════════════════════════════════════════════════════════════════

def main(args):
    levels = [int(x) for x in args.levels.split(",")]
    print(f"=== Rasterized-interference smoke test (levels={levels}) ===\n")
    print("[1] States")
    state = pick_state(args.states_dir, args.state_id)
    if state is None:
        sys.exit(1)

    print("\n[2] Constructed variants + rasterized cards")
    docs = check_variants(args.variants_dir, state, levels)
    if not docs:
        sys.exit(1)

    if args.skip_online:
        print("\n[3] Online checks SKIPPED (--skip_online)")
    else:
        print(f"\n[3] Online checks against port {args.port}")
        check_online(state, docs, args.variants_dir, levels, args.model, args.port,
                     template=args.force_template)

    print(f"\n=== summary: {N_PASS} pass, {N_WARN} warn, {N_FAIL} fail ===")
    sys.exit(1 if N_FAIL else 0)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--states_dir", required=True)
    p.add_argument("--variants_dir", required=True)
    p.add_argument("--state_id", default=None)
    p.add_argument("--model", default=os.path.join(os.environ.get("MODELS_DIR", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "models"))), "Qwen3.5-9B"))
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--levels", default="0,1,2,4,8")
    p.add_argument("--force_template", choices=["thought_action", "action_only"],
                   default="thought_action",
                   help="must match run_probe.py --force_template")
    p.add_argument("--skip_online", action="store_true",
                   help="only run the offline construction checks")
    main(p.parse_args())
