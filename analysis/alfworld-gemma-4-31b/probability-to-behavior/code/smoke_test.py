"""
Smoke test — verify the probability-to-behavior chain on ONE state before
committing to the full probe / rollout run.

Offline checks (no server needed):
  1. states + manifest exist; THE PREMISE: the competitor ("distractor") is in
     the state's admissible_commands (executable now) and is not the correct
     action; the correct action is admissible too
  2. text variants exist for every requested k: keyword count == k, literal
     action count == k, word count constant across k
  3. structural AND rasterized cards exist for every level, matching counts
  4. run_rollout.py's history parsing reproduces step_idx actions (the replay
     the rollout depends on)

Online checks (needs a vLLM server on --port):
  5. forced decoding k=0 vs k=kmax in all three representations; the full
     candidate-set distribution normalises (text, k=0)
  6. free generation parses; whether it picks the EXECUTABLE competitor is
     reported (that is the behaviour this experiment measures, so even one
     state gives a first signal)

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
# Offline
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
                   f"competitor='{state['distractor_action']}' "
                   f"keyword='{state['distractor_keyword']}' "
                   f"candidates={state['n_candidates']}")
    admissible = state["admissible_commands"]
    if state["correct_action"] in admissible:
        report("PASS", "correct action is admissible")
    else:
        report("FAIL", "correct action NOT admissible — distribution ill-defined")
    if state["distractor_action"] in admissible:
        report("PASS", "competitor IS admissible now (executable wrong action — "
                       "the premise of this experiment)")
    else:
        report("FAIL", "competitor NOT admissible — these are lexical-interference "
                       "states, not probability-to-behavior states; rerun "
                       "extract_behavior_states.py")
    if state.get("competitor_admissible") is True:
        report("PASS", "competitor_admissible flag set")
    else:
        report("FAIL", "competitor_admissible flag missing/false")
    if state["distractor_action"] != state["correct_action"]:
        report("PASS", "competitor differs from the correct action")
    else:
        report("FAIL", "competitor == correct action")
    return state


def check_variants(variants_dir, state, levels):
    sid = state["state_id"]
    keyword = state["distractor_keyword"]
    action = state["distractor_action"]
    text_dir = Path(variants_dir) / "text" / sid

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
        kc, ac = lex.count_word(doc, keyword), doc.count(action)
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
        report("FAIL", f"word count varies across k: {sorted(word_counts)}")

    counts = {"structural": {}, "rasterized": {}}
    for k in levels:
        for style, subdir in (("structural", "visual"), ("rasterized", "rasterized")):
            card_dir = Path(variants_dir) / subdir / sid / f"level_{k}"
            cards = sorted(card_dir.glob("card*.png")) if card_dir.exists() else []
            counts[style][k] = len(cards)
            if not cards:
                report("FAIL", f"no {style} cards at {card_dir}")
    for style, cc in counts.items():
        if all(v > 0 for v in cc.values()) and len(set(cc.values())) == 1:
            report("PASS", f"{style} cards consistent: {cc[levels[0]]} per level")
    return docs


def check_rollout_replay(state):
    """run_rollout.py replays len(history_actions(state)) == step_idx actions."""
    import run_rollout as rr
    acts = rr.history_actions(state)
    if len(acts) == state["step_idx"]:
        report("PASS", f"rollout replay: {len(acts)} history actions == step_idx")
    else:
        report("FAIL", f"rollout replay: {len(acts)} history actions != "
                       f"step_idx {state['step_idx']}")


# ══════════════════════════════════════════════════════════════════════
# Online
# ══════════════════════════════════════════════════════════════════════

def check_online(state, docs, variants_dir, levels, model, port,
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
    candidates = list(state["candidate_actions"])
    k0, kmax = levels[0], levels[-1]

    for rep in ("text", "rasterized", "structural"):
        print(f"\n  --- {rep.upper()} condition ---")
        results = {}
        for k in (k0, kmax):
            if rep == "text":
                if not docs.get(k):
                    report("FAIL", f"text k={k}: document missing")
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
            if lp_d.get("stem_prefix_ok") is False or lp_d.get("tokenization_length_ok") is False:
                report("FAIL", f"{rep} k={k}: forced-stem/tokenisation alignment broken")
                continue
            if not lp_d["complete"] or lp_d["action_logprob"] is None:
                report("FAIL", f"{rep} k={k}: forced decoding of competitor incomplete "
                               f"({lp_d['matched_tokens']}/{lp_d['n_tokens']})")
                continue
            if not lp_c["complete"] or lp_c["action_logprob"] is None:
                report("FAIL", f"{rep} k={k}: forced decoding of correct incomplete")
                continue
            results[k] = {"dist": lp_d["action_logprob"], "corr": lp_c["action_logprob"]}
            print(f"    k={k}: logP(competitor)={lp_d['action_logprob']:.4f}  "
                  f"logP(correct)={lp_c['action_logprob']:.4f}")
        if len(results) == 2:
            report("PASS", f"{rep}: forced decoding works at k={k0} and k={kmax}")
            dd = results[kmax]["dist"] - results[k0]["dist"]
            if rep == "text" and dd <= 0:
                report("WARN", f"text logP(competitor) did NOT increase k{k0}→k{kmax} "
                               f"(Δ={dd:+.4f}) on this state — judge on the full run")
            else:
                report("PASS", f"{rep}: ΔlogP(competitor) k{k0}→k{kmax} = {dd:+.4f}")

        if rep == "text" and docs.get(k0):
            messages = pl.build_messages(state, "text", skill_text=docs[k0])
            scores, _ = pl.score_candidates(messages, candidates, model=model,
                                            template=template)
            stats = pl.distribution_stats(scores, correct, distractor)
            if stats.get("n_scored") == len(candidates):
                psum = sum(stats["p_sum"].values())
                report("PASS", f"text: full candidate set scored, Σp={psum:.6f}, "
                               f"argmax='{stats['argmax_sum']}'")
                if abs(psum - 1.0) > 1e-6:
                    report("FAIL", "normalised distribution does not sum to 1")
            else:
                report("FAIL", f"text: only {stats.get('n_scored')}/{len(candidates)} "
                               f"candidates scored")

        # Free generation at kmax — the behavioural half of the bridge.
        if docs.get(kmax):
            if rep == "text":
                messages = pl.build_messages(state, "text", skill_text=docs[kmax])
            else:
                subdir = "rasterized" if rep == "rasterized" else "visual"
                card_dir = Path(variants_dir) / subdir / sid / f"level_{kmax}"
                cards = sorted(card_dir.glob("card*.png"))
                if not cards:
                    continue
                messages = pl.build_messages(state, rep,
                                             card_paths=[str(c) for c in cards])
            text, _ = pl.generate_action(messages, model=model, max_tokens=256)
            gen = pl.summarise_generation(text, correct, distractor, candidates)
            if not gen["chosen_action"]:
                report("WARN", f"{rep}: free generation did not parse; raw: {text[:120]!r}")
            elif gen["chose_distractor"]:
                report("PASS", f"{rep} k={kmax}: free generation EXECUTED-able chose "
                               f"the competitor '{gen['chosen_action']}' — the "
                               f"behaviour this experiment measures (single state!)")
            elif gen["chose_correct"]:
                report("PASS", f"{rep} k={kmax}: free generation chose the correct "
                               f"action on this state")
            else:
                report("WARN", f"{rep} k={kmax}: free generation chose a third action "
                               f"'{gen['chosen_action']}'")


def main(args):
    levels = [int(x) for x in args.levels.split(",")]
    print(f"=== Probability-to-behavior smoke test (levels={levels}) ===\n")
    print("[1] States (premise: the competitor is executable)")
    state = pick_state(args.states_dir, args.state_id)
    if state is None:
        sys.exit(1)

    print("\n[2] Constructed variants + cards")
    docs = check_variants(args.variants_dir, state, levels)
    if not docs:
        sys.exit(1)

    print("\n[3] Rollout replay bookkeeping")
    check_rollout_replay(state)

    if args.skip_online:
        print("\n[4] Online checks SKIPPED (--skip_online)")
    else:
        print(f"\n[4] Online checks against port {args.port}")
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
                   default="thought_action")
    p.add_argument("--skip_online", action="store_true")
    main(p.parse_args())
