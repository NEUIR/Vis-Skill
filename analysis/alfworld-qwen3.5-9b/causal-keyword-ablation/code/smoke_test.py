"""
Smoke test — verify the causal-keyword-ablation chain on ONE state before
committing to the full probe run.

Offline checks (no server needed):
  1. states + manifest exist; the distractor is NOT admissible (as designed)
  2. for every requested level k, all three document versions exist and:
       original         keyword count == k
       counterfactual   keyword count == 0, word count == original,
                        exactly k lines differ
       matched_control  control-word count k -> 0, keyword count STILL == k,
                        word count == original
  3. cards exist for every version (one per skill part) in the rendered styles

Online checks (needs a vLLM server on --port):
  4. forced decoding of the distractor works for original AND counterfactual
     in the text and structural conditions, and reports the paired drop
     ΔlogP(distractor) = original − counterfactual (a non-drop is WARN —
     it can be a genuine null result on one state)
  5. free generation parses

Usage:
    python smoke_test.py --states_dir ../states --variants_dir ../variants \
        [--state_id ...] [--port 8000] [--skip_online] [--levels 4,8]
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


def differing_lines(a, b):
    la, lb = a.split("\n"), b.split("\n")
    if len(la) != len(lb):
        return None
    return sum(1 for x, y in zip(la, lb) if x != y)


# ══════════════════════════════════════════════════════════════════════
# Offline
# ══════════════════════════════════════════════════════════════════════

def pick_state(states_dir, state_id, variants_dir, levels):
    states_dir = Path(states_dir)
    manifest = states_dir / "manifest.json"
    if not manifest.exists():
        report("FAIL", f"no manifest at {manifest}")
        return None
    entries = json.loads(manifest.read_text(encoding="utf-8"))["states"]
    if state_id:
        entry = next((e for e in entries if e["state_id"] == state_id), None)
        if entry is None:
            report("FAIL", f"state_id {state_id} not in manifest")
            return None
        candidates = [entry]
    else:
        # Prefer a state whose counterfactual documents are all built.
        def built(e):
            d = Path(variants_dir) / "text" / e["state_id"]
            return all((d / f"{v}_k{k}.md").exists()
                       for v in ("original", "counterfactual", "matched_control")
                       for k in levels)
        candidates = [e for e in entries if built(e)] or entries
        entry = next((e for e in candidates
                      if e.get("distractor_keyword") in ("microwave", "fridge")),
                     candidates[0])
    sf = states_dir / f"{entry['state_id']}.json"
    if not sf.exists():
        report("FAIL", f"state file missing: {sf}")
        return None
    state = json.loads(sf.read_text(encoding="utf-8"))
    report("PASS", f"state {state['state_id']}: correct='{state['correct_action']}' "
                   f"distractor='{state['distractor_action']}' "
                   f"keyword='{state['distractor_keyword']}'")
    if state["distractor_action"] not in state["admissible_commands"]:
        report("PASS", "distractor action is NOT admissible now (as designed)")
    else:
        report("FAIL", "distractor is admissible — wrong state set for this experiment?")
    return state


def check_versions(variants_dir, state, levels, styles):
    sid = state["state_id"]
    keyword = state["distractor_keyword"]
    text_dir = Path(variants_dir) / "text" / sid

    cmeta_file = Path(variants_dir) / "construction_meta.json"
    cmeta = {}
    if cmeta_file.exists():
        for m in json.loads(cmeta_file.read_text(encoding="utf-8"))["states"]:
            if m["state_id"] == sid:
                cmeta = m.get("levels", {})

    ok = True
    for k in levels:
        docs = {}
        for v in ("original", "counterfactual", "matched_control"):
            f = text_dir / f"{v}_k{k}.md"
            if not f.exists():
                report("FAIL", f"missing {f}")
                ok = False
                continue
            docs[v] = f.read_text(encoding="utf-8")
        if len(docs) < 2 or "original" not in docs:
            continue
        orig = docs["original"]
        kc = lex.count_word(orig, keyword)
        if kc != k:
            report("FAIL", f"k={k}: original keyword count is {kc}, expected {k}")
            ok = False

        if "counterfactual" in docs:
            cf = docs["counterfactual"]
            if lex.count_word(cf, keyword) != 0:
                report("FAIL", f"k={k}: counterfactual still contains '{keyword}'")
                ok = False
            if lex.word_count(cf) != lex.word_count(orig):
                report("FAIL", f"k={k}: counterfactual word count changed "
                               f"({lex.word_count(orig)} -> {lex.word_count(cf)})")
                ok = False
            dl = differing_lines(orig, cf)
            if dl != k:
                report("WARN" if dl is not None else "FAIL",
                       f"k={k}: counterfactual differs in {dl} lines, expected exactly {k}")
            # Note: the action VERB stays by design — only the keyword is
            # replaced ("open microwave 1" -> "open appliance 1"), matching the
            # user-level description of the counterfactual exactly.

        vm = cmeta.get(str(k), {}).get("versions", {}).get("matched_control", {})
        if "matched_control" in docs:
            mc = docs["matched_control"]
            ctrl = vm.get("control_word")
            if not ctrl:
                report("WARN", f"k={k}: no control word recorded in construction_meta")
            else:
                before, after = lex.count_word(orig, ctrl), lex.count_word(mc, ctrl)
                if before != k or after != 0:
                    report("FAIL", f"k={k}: control word '{ctrl}' count {before}->{after}, "
                                   f"expected {k}->0")
                    ok = False
                else:
                    report("PASS", f"k={k}: control word '{ctrl}' (tier {vm.get('tier')}) "
                                   f"{k} -> 0, replacement '{vm.get('control_replacement')}'")
            if lex.count_word(mc, keyword) != k:
                report("FAIL", f"k={k}: matched_control changed the keyword count "
                               f"({lex.count_word(mc, keyword)} != {k})")
                ok = False
            if lex.word_count(mc) != lex.word_count(orig):
                report("FAIL", f"k={k}: matched_control word count changed")
                ok = False

        # cards
        for style, subdir in (("structural", "visual"), ("rasterized", "rasterized")):
            if style not in styles:
                continue
            for v in docs:
                card_dir = Path(variants_dir) / subdir / sid / f"{v}_k{k}"
                n = len(sorted(card_dir.glob("card*.png"))) if card_dir.exists() else 0
                if n == 0:
                    report("FAIL", f"k={k} {v}: no {style} cards at {card_dir}")
                    ok = False
    if ok:
        report("PASS", f"all version documents consistent for levels {levels}")
    return ok


# ══════════════════════════════════════════════════════════════════════
# Online
# ══════════════════════════════════════════════════════════════════════

def check_online(state, variants_dir, levels, reps, model, port,
                 template="thought_action"):
    import requests
    import probe_lib as pl

    try:
        r = requests.get(f"http://localhost:{port}/health", timeout=5)
        if r.status_code != 200:
            report("FAIL", f"vLLM server on port {port} not healthy")
            return
    except Exception as e:
        report("FAIL", f"vLLM server on port {port} not reachable ({e})")
        return
    pl.set_endpoint(port)

    sid = state["state_id"]
    correct = lex.normalize_action(state["correct_action"])
    distractor = lex.normalize_action(state["distractor_action"])
    kmax = levels[-1]

    for rep in reps:
        print(f"\n  --- {rep.upper()} condition (k={kmax}) ---")
        lp = {}
        for version in ("original", "counterfactual", "matched_control"):
            md = Path(variants_dir) / "text" / sid / f"{version}_k{kmax}.md"
            if not md.exists():
                report("FAIL", f"{rep}/{version}: document missing")
                continue
            doc = md.read_text(encoding="utf-8")
            if rep == "text":
                messages = pl.build_messages(state, "text", skill_text=doc)
            else:
                subdir = "rasterized" if rep == "rasterized" else "visual"
                card_dir = Path(variants_dir) / subdir / sid / f"{version}_k{kmax}"
                cards = sorted(card_dir.glob("card*.png"))
                if not cards:
                    report("FAIL", f"{rep}/{version}: no cards at {card_dir}")
                    continue
                messages = pl.build_messages(state, rep, card_paths=[str(c) for c in cards])
            two, _ = pl.score_candidates(messages, [distractor, correct],
                                         model=model, template=template)
            d, c = two[distractor], two[correct]
            if d.get("stem_prefix_ok") is False or d.get("tokenization_length_ok") is False:
                report("FAIL", f"{rep}/{version}: forced-stem/tokenisation alignment broken")
                continue
            if not d["complete"] or d["action_logprob"] is None:
                report("FAIL", f"{rep}/{version}: forced decoding incomplete "
                               f"({d['matched_tokens']}/{d['n_tokens']})")
                continue
            lp[version] = d["action_logprob"]
            print(f"    {version:<16} logP(distractor)={d['action_logprob']:.4f}  "
                  f"logP(correct)={c['action_logprob'] if c['complete'] else float('nan'):.4f}")

        if "original" in lp and "counterfactual" in lp:
            report("PASS", f"{rep}: forced decoding works for original + counterfactual")
            d_cf = lp["original"] - lp["counterfactual"]
            if d_cf > 0:
                report("PASS", f"{rep}: removing the keyword DROPPED logP(distractor) "
                               f"by {d_cf:+.4f} nats on this state")
            else:
                report("WARN", f"{rep}: removing the keyword did NOT drop "
                               f"logP(distractor) on this state (Δ={d_cf:+.4f}) — "
                               f"possible null result, judge on the full run")
            if "matched_control" in lp:
                d_mc = lp["original"] - lp["matched_control"]
                report("PASS" if abs(d_mc) < abs(d_cf) else "WARN",
                       f"{rep}: control-word removal Δ={d_mc:+.4f} vs keyword Δ={d_cf:+.4f} "
                       f"(DiD={d_cf - d_mc:+.4f} on this state)")

    # Free generation parse (text, counterfactual, kmax)
    md = Path(variants_dir) / "text" / sid / f"counterfactual_k{kmax}.md"
    if md.exists():
        messages = pl.build_messages(state, "text",
                                     skill_text=md.read_text(encoding="utf-8"))
        text, _ = pl.generate_action(messages, model=model, max_tokens=256)
        gen = pl.summarise_generation(text, correct, distractor,
                                      state["candidate_actions"])
        if gen["chosen_action"]:
            report("PASS", f"free generation parses: '{gen['chosen_action']}'")
        else:
            report("WARN", f"free generation did not parse; raw: {text[:120]!r}")


def main(args):
    levels = [int(x) for x in args.levels.split(",")]
    reps = [r.strip() for r in args.representations.split(",") if r.strip()]
    styles = ["structural"] if "structural" in reps else []
    styles += ["rasterized"] if "rasterized" in reps else []
    print(f"=== Causal-keyword-ablation smoke test (levels={levels}, reps={reps}) ===\n")
    print("[1] States")
    state = pick_state(args.states_dir, args.state_id, args.variants_dir, levels)
    if state is None:
        sys.exit(1)

    print("\n[2] Counterfactual documents + cards")
    if not check_versions(args.variants_dir, state, levels, styles):
        sys.exit(1)

    if args.skip_online:
        print("\n[3] Online checks SKIPPED (--skip_online)")
    else:
        print(f"\n[3] Online checks against port {args.port}")
        check_online(state, args.variants_dir, levels, reps, args.model, args.port,
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
    p.add_argument("--levels", default="4,8")
    p.add_argument("--representations", default="text,structural")
    p.add_argument("--force_template", choices=["thought_action", "action_only"],
                   default="thought_action")
    p.add_argument("--skip_online", action="store_true")
    main(p.parse_args())
