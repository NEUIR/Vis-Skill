"""
Smoke test — verify the whole measurement chain on ONE state before committing
to the full probe run.

Offline checks (no server needed):
  1. states + manifest exist, one state can be loaded
  2. text variants exist for every requested k and
       keyword count == k, phrase count == k, word count constant across k
  3. visual cards exist (one per skill part per level)

Online checks (needs a vLLM server on --port):
  4. TEXT  k=0 vs k=kmax: forced-decoding logP(distractor) / logP(correct),
     full candidate-set distribution sums to 1, entropy finite
  5. VISUAL k=0 vs k=kmax: same, i.e. the fixed continue_final_message +
     prompt_logprobs path produces usable, level-dependent numbers for the
     image condition (the bug class that broke the visual runs of an earlier pilot analysis)
  6. free generation parses into a valid action

Soft failures (e.g. "distractor logprob did not increase with k") print WARN —
that can be a genuine null result. Hard failures (missing files, dead server,
unusable logprobs) print FAIL and set the exit code.

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
        # Prefer a state whose distractor keyword is a classic appliance word.
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
    phrase = state["distractor_phrase"]
    action = state["distractor_action"]
    text_dir = Path(variants_dir) / "text" / sid
    vis_dir = Path(variants_dir) / "visual" / sid

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
        wc = lex.word_count(doc)
        word_counts.add(wc)
        if kc != k:
            report("FAIL", f"k={k}: keyword '{keyword}' count is {kc}, expected {k}")
            ok = False
        # The literal action string is the other half of the manipulation. (The
        # contiguous "{verb} {noun}" phrase is NOT checked here: it only exists
        # for one-argument actions — "clean X with sinkbasin" interleaves the
        # object — so it would false-alarm on every clean/heat/cool distractor.)
        if ac != k:
            report("FAIL", f"k={k}: literal action '{action}' count is {ac}, expected {k}")
            ok = False
    if ok:
        report("PASS", f"keyword count == k for all levels {levels}")
    if len(word_counts) <= 1:
        report("PASS", f"word count constant across k ({word_counts})")
    else:
        report("FAIL", f"word count varies across k: {sorted(word_counts)} — "
                       f"length confound!")

    n_cards = {}
    for k in levels:
        card_dir = vis_dir / f"level_{k}"
        cards = sorted(card_dir.glob("card*.png")) if card_dir.exists() else []
        n_cards[k] = len(cards)
        if not cards:
            report("FAIL", f"no visual cards at {card_dir}")
    if all(v > 0 for v in n_cards.values()) and len(set(n_cards.values())) == 1:
        report("PASS", f"visual cards present and consistent: {n_cards[levels[0]]} per level")
    return docs, n_cards


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

    for rep in ("text", "visual"):
        print(f"\n  --- {rep.upper()} condition ---")
        results = {}
        for k in (k0, kmax):
            if rep == "text":
                if not docs.get(k):
                    report("FAIL", f"text k={k}: document missing, cannot probe")
                    continue
                messages = pl.build_messages(state, "text", skill_text=docs.get(k))
            else:
                card_dir = Path(variants_dir) / "visual" / sid / f"level_{k}"
                cards = sorted(card_dir.glob("card*.png"))
                if not cards:
                    report("FAIL", f"{rep} k={k}: no cards to send")
                    continue
                messages = pl.build_messages(state, "visual",
                                             card_paths=[str(c) for c in cards])

            # Score through the SAME path run_probe.py uses: score_candidates
            # derives the constant stem's tokenisation once and passes it down,
            # which is what makes `action_logprob` (the stem-excluded, primary
            # metric) available at all. Calling score_action directly without
            # stem_tokens leaves action_logprob None and the checks below can
            # never pass.
            two, _ = pl.score_candidates(messages, [distractor, correct],
                                         model=model, template=template)
            lp_d, lp_c = two[distractor], two[correct]
            if lp_d.get("stem_prefix_ok") is False or lp_c.get("stem_prefix_ok") is False:
                report("FAIL", f"{rep} k={k}: the forced stem is not a token prefix of "
                               f"the forced turn — logP(action) cannot be isolated, the "
                               f"primary metric would be None for every cell")
                continue
            if lp_d.get("tokenization_length_ok") is False:
                report("FAIL", f"{rep} k={k}: /tokenize and prompt_logprobs disagree on "
                               f"the token count — scores would be silently truncated")
                continue
            if not lp_d["complete"] or lp_d["action_logprob"] is None:
                report("FAIL", f"{rep} k={k}: forced decoding of distractor incomplete "
                               f"(matched {lp_d['matched_tokens']}/{lp_d['n_tokens']}, "
                               f"stem_prefix_ok={lp_d['stem_prefix_ok']})")
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
                               f"to k={kmax} (Δ={dd:+.4f}) — possible null result, "
                               f"check on more states before concluding")
            else:
                report("PASS", f"{rep}: ΔlogP(distractor) k{k0}→k{kmax} = {dd:+.4f}")
        elif results:
            report("WARN", f"{rep}: only {len(results)}/2 levels usable")

        # Full candidate-set distribution at k=0 (normalisation sanity)
        if rep == "text" and docs.get(k0):
            messages = pl.build_messages(state, "text", skill_text=docs[k0])
            scores, _ = pl.score_candidates(messages, candidates, model=model,
                                            template=template)
            stats = pl.distribution_stats(scores, correct, distractor)
            if stats.get("n_scored") == len(candidates):
                psum = sum(stats["p_sum"].values())
                report("PASS", f"text: full candidate set scored "
                               f"({stats['n_scored']}/{len(candidates)}), "
                               f"Σp={psum:.6f}, H={stats['entropy_sum']:.3f} nats, "
                               f"argmax='{stats['argmax_sum']}'")
                if abs(psum - 1.0) > 1e-6:
                    report("FAIL", "normalised distribution does not sum to 1")
            else:
                report("FAIL", f"text: only {stats.get('n_scored')}/{len(candidates)} "
                               f"candidates scored; incomplete: "
                               f"{stats.get('incomplete_candidates')}")

    # Free generation parse check (text, k=kmax)
    if docs.get(kmax):
        messages = pl.build_messages(state, "text", skill_text=docs[kmax])
        text, _ = pl.generate_action(messages, model=model, max_tokens=256)
        gen = pl.summarise_generation(text, correct, distractor, candidates)
        if gen["chosen_action"]:
            report("PASS", f"free generation parses: '{gen['chosen_action']}' "
                           f"(correct={gen['chose_correct']}, "
                           f"distractor={gen['chose_distractor']})")
        else:
            report("WARN", f"free generation did not parse an action; raw: {text[:120]!r}")


# ══════════════════════════════════════════════════════════════════════

def main(args):
    levels = [int(x) for x in args.levels.split(",")]
    print(f"=== Lexical-interference smoke test (levels={levels}) ===\n")
    print("[1] States")
    state = pick_state(args.states_dir, args.state_id)
    if state is None:
        sys.exit(1)

    print("\n[2] Constructed variants")
    docs, n_cards = check_variants(args.variants_dir, state, levels)
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
