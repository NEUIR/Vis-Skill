"""
Step 1 — extract single-step decision states whose distractor is a REAL,
CURRENTLY-EXECUTABLE competing action (the probability-to-behavior bridge).

Difference from ../../lexical-interference/code/extract_decision_states.py
(everything else — replay machinery, clean history, candidate sets, field
schema — is deliberately identical, so construct_variants.py / render_cards.py /
render_rasterized.py / run_probe.py work on these states unchanged):

    lexical-interference distractor:  plausible but NOT admissible right now
                                      ("open microwave 1" while the microwave is
                                      out of reach) — the model can only show a
                                      probability shift, never execute it.

    this experiment's competitor:     admissible (the environment WOULD execute
                                      it) but wrong/premature for the current
                                      task step — e.g. correct = "take apple 1
                                      from countertop 1" while "open microwave
                                      1" is also executable right now.

That closes the last gap in the chain: if the skill keyword frequency raises
P(competitor) AND the free generation actually starts choosing an action the
environment will carry out, then "Skill keyword frequency ↑ → wrong-action preference ↑ →
the model actually executes the wrong action" is fully connected — and run_rollout.py can price the
detour in extra steps / task failures.

Selection constraints for the competitor (mirror image of the LI distractor):
  (a) semantically plausible but premature for this task family, from two
      families — premature INTERACTION (open the heating/cooling appliance,
      heat/cool/clean prematurely, use lamp, open storage; same families the
      LI distractor uses, but requiring it to be admissible right now) and
      premature NAVIGATION (`go to {family appliance}` — admissible from
      anywhere in ALFWorld, and a real detour when the plan does not call
      for it yet; interaction candidates are preferred when both exist);
  (b) IS present in this state's admissible_commands (the LI extractor
      rejected exactly these);
  (c) not the correct action itself. Unlike the LI distractor it MAY share
      the correct verb — `go to fridge 1` while the correct step is
      `go to countertop 1` is a wrong action even though both navigate;
  (d) its keyword occurs in the skill corpus, NOT in the task query and NOT in
      the correct action — so any later reproduction is skill-attributable;
  (e) for examine tasks while holding the target, `use {lamp}` is excluded:
      it could complete the task immediately and would not be a wrong action.

Per decision state we keep the same schema as the LI states, with
`distractor_action` = the admissible competitor, `competitor_admissible: true`
and the audit trail of rejected alternatives. States whose decision point
offers NO admissible competitor are skipped (counted in the manifest).

Source pool and replay: identical to the LI extractor — games both main
experiments solved, replayed inside the real ALFWorld env to recover
`info["admissible_commands"]` at every step.
"""
import os
import re
import json
import yaml
import argparse
from pathlib import Path
from collections import Counter, defaultdict

from prompts import SYSTEM_PROMPT
import skill_lexicon as lex

os.environ.setdefault("ALFWORLD_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "data", "alfworld"))

CODE_DIR = Path(__file__).resolve().parent

META_COMMANDS = {"look", "inventory", "help", "wait"}
_EXAMINE_RE = re.compile(r"^examine\s", re.I)

FAILURE_FEEDBACK = "nothing happens"


# ══════════════════════════════════════════════════════════════════════
# Loading the two main experiments  (identical to the LI extractor)
# ══════════════════════════════════════════════════════════════════════

def load_run(results_dir):
    out = {}
    results_dir = Path(results_dir)
    for f in sorted(results_dir.glob("idx_*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[WARN] cannot read {f}: {e}")
            continue
        name = d.get("name", "")
        if not name:
            continue
        reward = d.get("reward", 0)
        if isinstance(reward, list):
            reward = reward[-1] if reward else 0
        success = reward in (True, 1)

        responses = [m["content"] for m in d.get("messages", [])
                     if m.get("role") == "assistant" and isinstance(m.get("content"), str)]
        steps = []
        for resp in responses:
            action = lex.parse_action_from_response(resp)
            if not action:
                continue
            steps.append({"action": action, "response": _trim_response(resp)})

        rec = {
            "game_name": name,
            "query": d.get("query", ""),
            "success": success,
            "n_steps": len(steps),
            "steps": steps,
            "relevant_skill_names": d.get("relevant_skill_names", []),
            "source_file": str(f),
        }
        prev = out.get(name)
        if prev is None or (rec["success"], rec["n_steps"]) > (prev["success"], prev["n_steps"]):
            out[name] = rec
    return out


def _trim_response(resp):
    thought = ""
    m = re.search(r"Thought:\s*(.+?)(?:\n|$)", resp, re.I)
    if m:
        thought = m.group(1).strip()
    action = lex.parse_action_from_response(resp)
    if not thought:
        thought = "I will take the next step of the plan."
    return f"Thought: {thought}\nAction: {action}"


def load_skill_ranking(path, top_k=5):
    by_name = {}
    if not (path and os.path.exists(path)):
        print(f"[WARN] skill ranking not found: {path}")
        return by_name
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for inst in data.get("instances", []):
        ranking = sorted(inst.get("skill_ranking", []),
                         key=lambda s: s.get("relevance_score", 0), reverse=True)
        name = inst.get("task_name", "")
        if name:
            by_name[name] = [s["skill_name"] for s in ranking[:top_k]]
    print(f"[INFO] skill ranking: {len(by_name)} instances, top-{top_k} each")
    return by_name


# ══════════════════════════════════════════════════════════════════════
# Environment  (identical to the LI extractor)
# ══════════════════════════════════════════════════════════════════════

def build_env(config_path, split):
    import alfworld  # noqa: F401
    from alfworld.agents.environment import get_environment

    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    env = get_environment(config["env"]["type"])(config, train_eval=split)
    env = env.init_env(batch_size=1)
    return env, config


def game_name_of(gamefile):
    return "/".join(str(gamefile).split("/")[-3:-1])


def process_ob(ob):
    if ob.startswith("You arrive at loc "):
        ob = ob[ob.find(". ") + 2:]
    return ob


def initial_observation(obs_raw):
    return "\n".join(obs_raw.split("\n\n")[1:]).strip() or obs_raw.strip()


# ══════════════════════════════════════════════════════════════════════
# Candidate action set  (identical to the LI extractor)
# ══════════════════════════════════════════════════════════════════════

def filter_candidates(admissible, keep_examine=False):
    out = []
    for a in admissible:
        an = lex.normalize_action(a)
        if not an or an in META_COMMANDS:
            continue
        if not keep_examine and _EXAMINE_RE.match(an):
            continue
        if an not in out:
            out.append(an)
    return out


def cap_candidates(candidates, must_keep, max_candidates):
    kept = [c for c in must_keep if c in candidates]
    rest = [c for c in candidates if c not in kept]
    if len(kept) + len(rest) <= max_candidates:
        return kept + rest

    by_verb = defaultdict(list)
    for c in rest:
        by_verb[lex.action_verb(c)].append(c)
    for v in by_verb:
        by_verb[v].sort()
    verbs = sorted(by_verb)
    i = 0
    while len(kept) < max_candidates and any(by_verb[v] for v in verbs):
        v = verbs[i % len(verbs)]
        if by_verb[v]:
            kept.append(by_verb[v].pop(0))
        i += 1
    return kept[:max_candidates]


# ══════════════════════════════════════════════════════════════════════
# Competitor choice — THE difference from the LI extractor
# ══════════════════════════════════════════════════════════════════════

# ── premature-navigation family ──
# Interaction commands (`open microwave 1`) are admissible only when the agent
# stands at that receptacle, so a LI-style "premature interaction" competitor
# is rare — most decision points would yield no state at all. `go to {recep}`
# is admissible from ANYWHERE in ALFWorld, and walking to a task-family
# appliance before the plan needs it is exactly a "currently executable but
# wrong/premature" detour (the rollout stage prices its real cost in extra
# steps). These candidates are appended AFTER the interaction family, so a
# genuinely premature interaction is still preferred whenever one exists.
_GOTO_FAMILY_ORDER = {
    "pick_heat":  ["heat", "clean", "cool", "lamp", "storage"],
    "pick_cool":  ["cool", "heat", "clean", "lamp", "storage"],
    "pick_clean": ["clean", "heat", "cool", "lamp", "storage"],
    "pick_place": ["heat", "cool", "clean", "lamp", "storage"],
    "pick_two":   ["heat", "cool", "clean", "lamp", "storage"],
    "examine":    ["storage", "heat", "cool", "clean", "lamp"],
}
_GOTO_NOUNS = {
    "heat": lex.HEAT_APPLIANCES,
    "cool": lex.COOL_APPLIANCES,
    "clean": lex.CLEAN_STATIONS,
    "lamp": lex.LAMPS,
    "storage": ["cabinet", "drawer", "safe", "box", "laundryhamper"],
}


def propose_competitors(task_type, room_receptacles, seen_objects, held_object,
                        correct_action):
    """LI premature-interaction candidates + premature-navigation (go-to)
    candidates, interaction family first, deterministic order, deduplicated."""
    cands = lex.propose_distractors(
        task_type, room_receptacles, seen_objects, held_object)
    recep_by_noun = {}
    for r in room_receptacles:
        recep_by_noun.setdefault(lex.head_noun(r), []).append(r)
    correct_norm = lex.normalize_action(correct_action)
    seen = {lex.normalize_action(c["action"]) for c in cands}
    seen.add(correct_norm)
    for family in _GOTO_FAMILY_ORDER.get(task_type, _GOTO_FAMILY_ORDER["pick_place"]):
        for noun in _GOTO_NOUNS[family]:
            for recep in recep_by_noun.get(noun, []):
                action = f"go to {recep}"
                if lex.normalize_action(action) in seen:
                    continue
                seen.add(lex.normalize_action(action))
                cands.append({
                    "action": action,
                    "verb": "go_to",
                    "noun": noun,
                    "phrase": f"go to {noun}",
                    "kind": f"premature_goto_{family}",
                })
    return cands


def choose_competitor(candidates, admissible_norm, correct_action, query, corpus,
                      task_type=None, held_object=None, allow_noun_in_query=False):
    """First candidate that IS executable now but is not the correct next step.

    Mirrors skill_lexicon.choose_distractor constraint-for-constraint, with two
    deliberate differences:
      * the applicability test is INVERTED (must be admissible now);
      * the "different verb than correct" rule is dropped — an executable
        detour is a wrong action even when it shares the correct verb (e.g.
        `go to fridge 1` while the correct step is `go to countertop 1`).
        The keyword constraints still guarantee clean attribution: the
        competitor's keyword occurs neither in the correct action nor in the
        task query, and it occurs in the skill corpus.
    """
    correct_norm = lex.normalize_action(correct_action)
    q = (query or "").lower()
    rejects = []

    for c in candidates:
        noun, verb = c["noun"], c["verb"]
        c_norm = lex.normalize_action(c["action"])
        if task_type == "examine" and held_object and c["kind"] == "premature_use_lamp":
            # Holding the target object, `use {lamp}` would COMPLETE an examine
            # task — that is the goal action, not a wrong one.
            rejects.append((c["action"], "lamp_use_may_complete_task"))
            continue
        if c_norm == correct_norm:
            rejects.append((c["action"], "is_correct_action"))
            continue
        if c_norm not in admissible_norm:
            rejects.append((c["action"], "not_admissible_now"))
            continue
        if lex.count_word(correct_norm, noun) > 0:
            rejects.append((c["action"], "keyword_in_correct_action"))
            continue
        noun_in_query = lex.count_word(q, noun) > 0
        if noun_in_query and not allow_noun_in_query:
            rejects.append((c["action"], "keyword_in_task_query"))
            continue
        wc = corpus.word_count(noun)
        if wc <= 0:
            rejects.append((c["action"], "keyword_absent_from_skill_corpus"))
            continue
        mentioning = corpus.skills_mentioning(noun)
        hint = lex.VERB_TO_SKILL_HINT.get(verb)
        out = dict(c)
        out["corpus_word_count"] = wc
        out["corpus_phrase_count"] = corpus.phrase_count(c["phrase"])
        out["skill"] = hint if hint in mentioning else (mentioning[0] if mentioning else "")
        out["skills_mentioning_keyword"] = mentioning
        out["noun_in_query"] = noun_in_query
        out["rejected_candidates"] = rejects
        return out
    return None


# ══════════════════════════════════════════════════════════════════════
# Replay of one trajectory  (same loop as the LI extractor)
# ══════════════════════════════════════════════════════════════════════

def replay_trajectory(env, record, args, corpus, skills_by_name, obs_raw, info):
    game_name = record["game_name"]
    task_type = lex.get_task_type(game_name)
    init_obs = initial_observation(obs_raw)
    query = record["query"] or (init_obs.split("Your task is to: ")[-1].split("\n")[0].strip())
    room_receptacles = lex.receptacles_in(init_obs)

    history = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": init_obs},
    ]
    seen_objects = list(lex.objects_in(init_obs))
    held_object = None

    admissible = list(info.get("admissible_commands", [[]])[0])
    decision_points = []
    skipped = Counter()

    for t, step in enumerate(record["steps"]):
        action = step["action"]
        action_norm = lex.normalize_action(action)
        admissible_norm = [lex.normalize_action(a) for a in admissible]
        candidates = filter_candidates(admissible, keep_examine=args.keep_examine)

        snapshot = {
            "step_idx": t,
            "history": [dict(m) for m in history],
            "correct_action": action_norm,
            "raw_action": action,
            "admissible_commands": admissible_norm,
            "candidates": candidates,
            "room_receptacles": list(room_receptacles),
            "seen_objects": list(seen_objects),
            "held_object": held_object,
        }

        obs_list, _, dones, info = env.step([action])
        feedback = process_ob(obs_list[0])
        admissible = list(info.get("admissible_commands", [[]])[0])

        usable = True
        if t == 0:
            skipped["first_step"] += 1
            usable = False
        if feedback.strip().lower().startswith(FAILURE_FEEDBACK):
            skipped["action_failed"] += 1
            usable = False
        if action_norm not in snapshot["admissible_commands"]:
            skipped["correct_action_not_admissible"] += 1
            usable = False
        if len(snapshot["candidates"]) < args.min_candidates:
            skipped["too_few_candidates"] += 1
            usable = False
        if usable:
            snapshot["feedback"] = feedback
            decision_points.append(snapshot)

        history.append({"role": "assistant", "content": step["response"]})
        history.append({"role": "user", "content": f"Observation: {feedback}"})
        for noun, full in lex.parse_entities(feedback):
            if noun in lex.OBJECT_NOUNS and full not in seen_objects:
                seen_objects.append(full)
            if noun in lex.RECEPTACLE_NOUNS and full not in room_receptacles:
                room_receptacles.append(full)
        m = re.search(r"you pick up the ([a-z]+ \d+)", feedback, re.I)
        if m:
            held_object = m.group(1).lower()
        if re.search(r"you (?:move|put) the", feedback, re.I):
            held_object = None

        if dones[0]:
            break

    if decision_points and args.drop_last_step:
        decision_points = decision_points[:-1]

    selected = _sample_evenly(decision_points, args.max_states_per_traj)

    states = []
    for dp in selected:
        cands = propose_competitors(
            task_type, dp["room_receptacles"], dp["seen_objects"], dp["held_object"],
            dp["correct_action"])
        comp = choose_competitor(
            cands, dp["admissible_commands"], dp["correct_action"], query, corpus,
            task_type=task_type, held_object=dp["held_object"],
            allow_noun_in_query=args.allow_noun_in_query)
        if comp is None:
            skipped["no_admissible_competitor"] += 1
            continue

        # The competitor is admissible, so it is already among the candidates;
        # cap_candidates still guarantees both it and the correct action survive.
        candidates = list(dp["candidates"])
        comp_norm = lex.normalize_action(comp["action"])
        if comp_norm not in candidates:
            candidates.append(comp_norm)
        candidates = cap_candidates(
            candidates,
            must_keep=[dp["correct_action"], comp_norm],
            max_candidates=args.max_candidates,
        )

        admissible_alternatives = sorted({
            lex.normalize_action(c["action"]) for c in cands
            if lex.normalize_action(c["action"]) in dp["admissible_commands"]
            and lex.normalize_action(c["action"]) != comp_norm
        })

        state_id = f"{re.sub(r'[^A-Za-z0-9]+', '-', game_name.split('/')[0])[:60]}__step{dp['step_idx']}"
        states.append({
            "state_id": state_id,
            "game_name": game_name,
            "task_type": task_type,
            "query": query,
            "step_idx": dp["step_idx"],
            "total_steps": record["n_steps"],
            "history_messages": dp["history"],
            "observation": dp["history"][-1]["content"],
            "correct_action": dp["correct_action"],
            "correct_action_type": lex.action_verb(dp["correct_action"]),
            "correct_action_feedback": dp["feedback"],
            # ── the competitor lives in the distractor_* fields, so every
            # downstream script (construct/render/probe) works unchanged ──
            "distractor_action": comp_norm,
            "distractor_action_type": comp["verb"],
            "distractor_keyword": comp["noun"],
            "distractor_phrase": comp["phrase"],
            "distractor_kind": f"admissible_{comp['kind']}",
            "distractor_skill": comp["skill"],
            "distractor_keyword_corpus_count": comp["corpus_word_count"],
            "distractor_phrase_corpus_count": comp["corpus_phrase_count"],
            "distractor_keyword_in_query": comp["noun_in_query"],
            "skills_mentioning_keyword": comp["skills_mentioning_keyword"],
            "competitor_admissible": True,
            "admissible_competitor_alternatives": admissible_alternatives,
            "rejected_competitor_candidates": comp["rejected_candidates"][:20],
            "candidate_actions": candidates,
            "n_candidates": len(candidates),
            "admissible_commands": dp["admissible_commands"],
            "held_object": dp["held_object"],
            "relevant_skills": skills_by_name.get(game_name, record["relevant_skill_names"]),
            "success_in": record["success_in"],
            "replay_source": record["replay_source"],
        })
    return states, skipped


def _sample_evenly(items, k):
    if len(items) <= k:
        return items
    step = len(items) / k
    return [items[min(len(items) - 1, int(i * step))] for i in range(k)]


# ══════════════════════════════════════════════════════════════════════
# Main  (same enumeration-drift-tolerant replay loop as the LI extractor)
# ══════════════════════════════════════════════════════════════════════

def main(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    text_run = load_run(args.text_results_dir)
    vis_run = load_run(args.vis_results_dir)
    print(f"[INFO] text run: {len(text_run)} games "
          f"({sum(r['success'] for r in text_run.values())} solved)")
    print(f"[INFO] visual run: {len(vis_run)} games "
          f"({sum(r['success'] for r in vis_run.values())} solved)")

    names = sorted(set(text_run) | set(vis_run))
    pool = []
    for n in names:
        t, v = text_run.get(n), vis_run.get(n)
        t_ok = bool(t and t["success"])
        v_ok = bool(v and v["success"])
        keep = {"both": t_ok and v_ok, "text": t_ok, "visual": v_ok,
                "any": t_ok or v_ok}[args.require]
        if not keep:
            continue
        if args.replay_source == "text":
            rec = t if t_ok else v
        elif args.replay_source == "visual":
            rec = v if v_ok else t
        else:
            cands = [r for r in (t, v) if r and r["success"]]
            rec = min(cands, key=lambda r: r["n_steps"]) if cands else (t or v)
        rec = dict(rec)
        rec["success_in"] = {"text": t_ok, "visual": v_ok}
        rec["replay_source"] = "text" if rec.get("source_file", "").find("text_baseline") >= 0 else "visual"
        pool.append(rec)

    if args.max_games > 0:
        pool = pool[:args.max_games]
    targets = {r["game_name"]: r for r in pool}
    print(f"[INFO] require={args.require} replay_source={args.replay_source} "
          f"-> {len(targets)} trajectories to replay")

    corpus = lex.SkillCorpus(args.skills_dir)
    print(f"[INFO] skill corpus: {len(corpus)} SKILL.md documents")
    skills_by_name = load_skill_ranking(args.skill_ranking_path, top_k=args.top_k)

    env, _ = build_env(args.config, args.split)
    gamefiles = list(env.gamefiles)
    total_games = len(gamefiles)
    name_by_idx = {i: game_name_of(g) for i, g in enumerate(gamefiles)}
    print(f"[INFO] env enumerates {total_games} games; "
          f"{len(set(targets) & set(name_by_idx.values()))}/{len(targets)} targets found by name")

    all_states = []
    skipped_total = Counter()
    replayed = 0
    found = set()
    for idx in range(total_games):
        obs_raw, info = env.reset()
        got = game_name_of(info["extra.gamefile"][0])
        expected = name_by_idx.get(idx)
        if got != expected:
            print(f"[WARN] env reset order drifted at idx={idx}: expected {expected}, "
                  f"got {got}. Trusting the reset output over the gamefiles order.")
        if got not in targets or got in found:
            continue
        record = targets[got]
        try:
            states, skipped = replay_trajectory(
                env, record, args, corpus, skills_by_name, obs_raw[0], info)
        except Exception as e:
            print(f"[WARN] replay failed for {got}: {e}")
            skipped_total["replay_exception"] += 1
            found.add(got)
            continue
        replayed += 1
        found.add(got)
        skipped_total.update(skipped)
        all_states.extend(states)
        print(f"[{replayed}/{len(targets)}] {got}: {len(states)} states")
        if len(found) == len(targets):
            print(f"[INFO] all {len(targets)} targets replayed; stopping at idx={idx}")
            break

    env.close()

    seen_ids = Counter()
    for s in all_states:
        seen_ids[s["state_id"]] += 1
        if seen_ids[s["state_id"]] > 1:
            s["state_id"] = f"{s['state_id']}-r{seen_ids[s['state_id']]}"

    for s in all_states:
        (out_dir / f"{s['state_id']}.json").write_text(
            json.dumps(s, indent=2, ensure_ascii=False), encoding="utf-8")

    manifest = {
        "total_states": len(all_states),
        "trajectories_replayed": replayed,
        "experiment": "probability-to-behavior",
        "note": "distractor_action is an ADMISSIBLE (currently executable) but "
                "wrong/premature competing action — competitor_admissible=true",
        "config": {
            "require": args.require,
            "replay_source": args.replay_source,
            "max_states_per_traj": args.max_states_per_traj,
            "max_candidates": args.max_candidates,
            "min_candidates": args.min_candidates,
            "keep_examine": args.keep_examine,
            "drop_last_step": args.drop_last_step,
            "allow_noun_in_query": args.allow_noun_in_query,
            "top_k_skills": args.top_k,
            "text_results_dir": args.text_results_dir,
            "vis_results_dir": args.vis_results_dir,
        },
        "skipped_decision_points": dict(skipped_total),
        "states": [
            {k: s[k] for k in (
                "state_id", "game_name", "task_type", "step_idx", "total_steps",
                "correct_action", "correct_action_type", "distractor_action",
                "distractor_action_type", "distractor_keyword", "distractor_phrase",
                "distractor_kind", "distractor_skill", "competitor_admissible",
                "distractor_keyword_corpus_count", "distractor_phrase_corpus_count",
                "distractor_keyword_in_query", "n_candidates", "relevant_skills",
                "success_in", "replay_source")}
            for s in all_states
        ],
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 68)
    print(f"Extracted {len(all_states)} decision states from {replayed} trajectories")
    print(f"  by task type        : {dict(Counter(s['task_type'] for s in all_states))}")
    print(f"  correct verb        : {dict(Counter(s['correct_action_type'] for s in all_states))}")
    print(f"  competitor verb     : {dict(Counter(s['distractor_action_type'] for s in all_states))}")
    print(f"  competitor keyword  : {dict(Counter(s['distractor_keyword'] for s in all_states))}")
    print(f"  competitor kind     : {dict(Counter(s['distractor_kind'] for s in all_states))}")
    print(f"  candidates/state    : mean "
          f"{sum(s['n_candidates'] for s in all_states) / max(1, len(all_states)):.1f}")
    print(f"  skipped points      : {dict(skipped_total)}")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--text_results_dir", required=True,
                   help="text-skill/results/alfworld/<model>/text_baseline")
    p.add_argument("--vis_results_dir", required=True,
                   help="vis-skill/results/alfworld/<model>/vis_baseline")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--skills_dir", required=True)
    p.add_argument("--skill_ranking_path",
                   default=str(CODE_DIR / "alfworld_skill_ranking_per_instance.json"))
    p.add_argument("--config", default=str(CODE_DIR / "base_config.yaml"))
    p.add_argument("--split", default="eval_out_of_distribution")
    p.add_argument("--require", choices=["both", "text", "visual", "any"], default="both")
    p.add_argument("--replay_source", choices=["text", "visual", "shorter"], default="text")
    p.add_argument("--max_states_per_traj", type=int, default=3)
    p.add_argument("--max_candidates", type=int, default=16)
    p.add_argument("--min_candidates", type=int, default=5)
    p.add_argument("--keep_examine", action="store_true")
    p.add_argument("--drop_last_step", action="store_true", default=True)
    p.add_argument("--no_drop_last_step", dest="drop_last_step", action="store_false")
    p.add_argument("--allow_noun_in_query", action="store_true")
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--max_games", type=int, default=-1)
    main(p.parse_args())
