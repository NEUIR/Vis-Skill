"""
Step 1 — extract single-step decision states from the two main experiments'
successful ALFWorld trajectories.

Unlike a script that only parses the saved trajectory files, this script **replays each
trajectory inside the real ALFWorld environment**, because the analysis needs the
one thing the saved trajectories do not contain: the set of *currently
executable actions* (`info["admissible_commands"]`). That set is what makes the
forced-decoding distribution well-defined (a candidate action set to normalise
over) and what lets us certify that a distractor is genuinely NOT applicable in
the current state.

Per decision state we keep:
  * the clean conditioning context — system prompt, initial room description,
    and the (assistant action, environment observation) pairs up to step t.
    "Clean" means: the procedural guidance / hint messages that the main runs'
    generated code injected are dropped, so the ONLY skill text in the probe
    prompt is the controlled document built in step 2.
  * `correct_action` — what the successful trajectory did at step t (and which
    the environment confirmed as executable at step t).
  * `candidate_actions` — the admissible commands, filtered and capped, always
    containing the correct action and the distractor.
  * `distractor_*` — a plausible-but-not-now action whose keyword occurs in the
    skill corpus, does not occur in the task query, and does not occur in the
    correct action (see skill_lexicon.choose_distractor).

Source pool: games that the *text* and the *visual* main experiment both solved
(`--require both`, the default). Both-solved keeps the pool free of any
per-modality selection bias — neither condition is handed states that only its
own modality could reach. `text` / `visual` / `any` are available too.

Note on indices: the two main runs enumerated the games in different os.walk
orders (132/134 of the shared idx values point at different games), so games are
matched by `game_name` throughout and `idx` is never used as a cross-run key.
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

# Admissible commands that are not real environment actions in the 9-action
# grammar the agent is prompted with.
META_COMMANDS = {"look", "inventory", "help", "wait"}
_EXAMINE_RE = re.compile(r"^examine\s", re.I)

FAILURE_FEEDBACK = "nothing happens"


# ══════════════════════════════════════════════════════════════════════
# Loading the two main experiments
# ══════════════════════════════════════════════════════════════════════

def load_run(results_dir):
    """-> {game_name: {success, steps, actions, thoughts, file}} for one run.

    Duplicate game_names (the same game reached under two env indices) keep the
    successful / longer record.
    """
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
    """Keep the Thought/Action pair only, so replayed history looks like a plain
    ReAct turn regardless of what the main run's generated code produced."""
    thought = ""
    m = re.search(r"Thought:\s*(.+?)(?:\n|$)", resp, re.I)
    if m:
        thought = m.group(1).strip()
    action = lex.parse_action_from_response(resp)
    if not thought:
        thought = "I will take the next step of the plan."
    return f"Thought: {thought}\nAction: {action}"


def load_skill_ranking(path, top_k=5):
    """-> {task_name: [skill_name, ...]} (top-k by relevance_score)."""
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
# Environment
# ══════════════════════════════════════════════════════════════════════

def build_env(config_path, split):
    import alfworld  # noqa: F401  (registers the envs)
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
    """Same slicing the main runs use: drop the engine banner paragraph."""
    return "\n".join(obs_raw.split("\n\n")[1:]).strip() or obs_raw.strip()


# ══════════════════════════════════════════════════════════════════════
# Candidate action set
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
    """Deterministically shrink the candidate set to `max_candidates`, keeping
    `must_keep` and otherwise round-robining over verbs so the surviving set
    stays verb-diverse (an entropy over a set dominated by 20 `go to` commands
    would be uninformative)."""
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
# Replay of one trajectory
# ══════════════════════════════════════════════════════════════════════

def replay_trajectory(env, record, args, corpus, skills_by_name, obs_raw, info):
    """Replay one successful trajectory and return its extracted states.

    The environment has just been reset onto this game; `obs_raw`/`info` are the
    reset outputs.
    """
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

        # Snapshot everything needed for a decision state BEFORE stepping.
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
            usable = False                       # no history yet -> trivial
        if feedback.strip().lower().startswith(FAILURE_FEEDBACK):
            skipped["action_failed"] += 1
            usable = False                       # the step did nothing
        if action_norm not in snapshot["admissible_commands"]:
            skipped["correct_action_not_admissible"] += 1
            usable = False                       # cannot normalise a distribution
        if len(snapshot["candidates"]) < args.min_candidates:
            skipped["too_few_candidates"] += 1
            usable = False
        if usable:
            snapshot["feedback"] = feedback
            decision_points.append(snapshot)

        # Advance the clean history and the object bookkeeping.
        history.append({"role": "assistant", "content": step["response"]})
        history.append({"role": "user", "content": f"Observation: {feedback}"})
        for noun, full in lex.parse_entities(feedback):
            if noun in lex.OBJECT_NOUNS and full not in seen_objects:
                seen_objects.append(full)
        m = re.search(r"you pick up the ([a-z]+ \d+)", feedback, re.I)
        if m:
            held_object = m.group(1).lower()
        if re.search(r"you (?:move|put) the", feedback, re.I):
            held_object = None

        if dones[0]:
            break

    # The last decision point of a successful run is the goal-completing action;
    # its distribution is dominated by the task text rather than by the skill.
    if decision_points and args.drop_last_step:
        decision_points = decision_points[:-1]

    selected = _sample_evenly(decision_points, args.max_states_per_traj)

    states = []
    for dp in selected:
        cands = lex.propose_distractors(
            task_type, dp["room_receptacles"], dp["seen_objects"], dp["held_object"])
        dist = lex.choose_distractor(
            cands, dp["admissible_commands"], dp["correct_action"], query, corpus,
            allow_noun_in_query=args.allow_noun_in_query)
        if dist is None:
            skipped["no_valid_distractor"] += 1
            continue

        candidates = list(dp["candidates"])
        if dist["action"] not in candidates:
            candidates.append(lex.normalize_action(dist["action"]))
        candidates = cap_candidates(
            candidates,
            must_keep=[dp["correct_action"], lex.normalize_action(dist["action"])],
            max_candidates=args.max_candidates,
        )

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
            "distractor_action": lex.normalize_action(dist["action"]),
            "distractor_action_type": dist["verb"],
            "distractor_keyword": dist["noun"],
            "distractor_phrase": dist["phrase"],
            "distractor_kind": dist["kind"],
            "distractor_skill": dist["skill"],
            "distractor_keyword_corpus_count": dist["corpus_word_count"],
            "distractor_phrase_corpus_count": dist["corpus_phrase_count"],
            "distractor_keyword_in_query": dist["noun_in_query"],
            "skills_mentioning_keyword": dist["skills_mentioning_keyword"],
            "candidate_actions": candidates,
            "n_candidates": len(candidates),
            "admissible_commands": dp["admissible_commands"],
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
# Main
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
        else:  # shorter successful trajectory
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
    idx_by_name = {}
    for i, n in name_by_idx.items():
        idx_by_name.setdefault(n, i)
    print(f"[INFO] env enumerates {total_games} games; "
          f"{len(set(targets) & set(idx_by_name))}/{len(targets)} targets found by name")

    # NOTE: the real env.reset() order drifts from the gamefiles enumeration
    # order (observed in our runs: every idx mismatched). Cutting the loop
    # at the gamefiles position of the last target would therefore silently
    # drop targets; iterate the full enumeration instead and stop early once
    # every target has been replayed.
    all_states = []
    skipped_total = Counter()
    replayed = 0
    found = set()
    for idx in range(total_games):
        obs_raw, info = env.reset()
        got = game_name_of(info["extra.gamefile"][0])
        expected = name_by_idx.get(idx)
        if got != expected:
            print(f"[WARN] env reset order drifted at idx={idx}: expected {expected}, got {got}. "
                  f"Trusting the reset output over the gamefiles order.")
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

    # De-duplicate state ids (two trials of the same task template can collide).
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
                "distractor_kind", "distractor_skill",
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
    print(f"  by task type      : {dict(Counter(s['task_type'] for s in all_states))}")
    print(f"  correct verb      : {dict(Counter(s['correct_action_type'] for s in all_states))}")
    print(f"  distractor verb   : {dict(Counter(s['distractor_action_type'] for s in all_states))}")
    print(f"  distractor keyword: {dict(Counter(s['distractor_keyword'] for s in all_states))}")
    print(f"  candidates/state  : mean "
          f"{sum(s['n_candidates'] for s in all_states) / max(1, len(all_states)):.1f}")
    print(f"  skipped points    : {dict(skipped_total)}")
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
    p.add_argument("--require", choices=["both", "text", "visual", "any"], default="both",
                   help="which main experiments must have solved the game")
    p.add_argument("--replay_source", choices=["text", "visual", "shorter"], default="text",
                   help="which successful trajectory to replay for the correct actions")
    p.add_argument("--max_states_per_traj", type=int, default=3)
    p.add_argument("--max_candidates", type=int, default=16,
                   help="cap on the forced-decoding candidate set per state")
    p.add_argument("--min_candidates", type=int, default=5)
    p.add_argument("--keep_examine", action="store_true",
                   help="keep 'examine X' admissible commands as candidates")
    p.add_argument("--drop_last_step", action="store_true", default=True)
    p.add_argument("--no_drop_last_step", dest="drop_last_step", action="store_false")
    p.add_argument("--allow_noun_in_query", action="store_true",
                   help="allow a distractor keyword that already occurs in the task query "
                        "(weakens skill attribution; off by default)")
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--max_games", type=int, default=-1)
    main(p.parse_args())
