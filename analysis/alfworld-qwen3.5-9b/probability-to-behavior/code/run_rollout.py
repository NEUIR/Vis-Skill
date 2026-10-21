"""
Step 3b (OPTIONAL) — roll the environment forward from each decision state to
price the competitor action in real task cost.

The probe measures single-step preference (probability + one free generation).
This script closes the chain end-to-end: probability shift → real execution → end-to-end task cost.

Per (state × k × representation × branch):
  branch = correct    step the trajectory's correct action, then continue with
                      the model's own greedy policy under the SAME skill
                      conditioning until done / budget
  branch = competitor step the admissible-but-wrong competitor action, then
                      continue identically

The comparison (analysis stage):
  * success rate within budget, correct vs competitor branch
  * extra steps to success on pairs where both succeed
  * "Nothing happened" counts and failure modes on the competitor branch

Environment mechanics: each worker process owns one ALFWorld env and walks the
game enumeration (env.reset() advances one game, exactly like
extract_behavior_states.py) until it reaches the task's game, then replays the
recorded history actions to the decision step. Every task therefore pays a
reset-walk over the enumeration — this stage is deliberately OPTIONAL and
bounded (--max_states, and run_full.sh gates it behind ROLLOUT=1).

Prompt/policy conventions (must match the probe, or the bridge breaks):
  * messages = clean history + the same skill-reference turn the probe cell
    used (text document / structural cards / rasterized cards) + the turns
    generated during this rollout;
  * the forced branch step is written into the history as
    "Thought: I will now perform the action.\nAction: {branch_action}"
    (probe_lib.FORCED_THOUGHT), so the policy continues from a context that
    looks exactly like its own previous outputs;
  * greedy generation (temperature 0), enable_thinking=False — the probe's
    free-generation settings.

Outputs: rollout_results/{state_id}__k{k}__{rep}__rollout_{branch}.json
         (resume-safe: existing files are skipped)
"""
import os
import re
import json
import queue
import argparse
import multiprocessing
from pathlib import Path
from tqdm import tqdm

import probe_lib as pl
import skill_lexicon as lex
from extract_behavior_states import (build_env, game_name_of, process_ob,
                                     CODE_DIR)

DEFAULT_MODEL = pl.DEFAULT_MODEL

CARD_DIRS = {"visual": "visual", "structural": "visual", "rasterized": "rasterized"}

_HISTORY_ACTION_RE = re.compile(r"Action:\s*(.+)", re.I)


# ══════════════════════════════════════════════════════════════════════
# One rollout
# ══════════════════════════════════════════════════════════════════════

def history_actions(state):
    """The actions executed BEFORE the decision step (for the env replay)."""
    acts = []
    for m in state["history_messages"]:
        if m.get("role") == "assistant" and isinstance(m.get("content"), str):
            mm = _HISTORY_ACTION_RE.search(m["content"])
            if mm:
                acts.append(mm.group(1).split("\n")[0].strip().strip("\"'*` "))
    return acts


def build_probe_messages(state, rep, variants_dir, level):
    """Reuse probe_lib.build_messages on a minimal state dict: clean history +
    the skill turn. Returns (messages, ok_reason)."""
    sid = state["state_id"]
    vdir = Path(variants_dir)
    if rep == "text":
        md = vdir / "text" / sid / f"level_{level}.md"
        if not md.exists():
            return None, f"missing {md}"
        msgs = pl.build_messages(state, "text",
                                 skill_text=md.read_text(encoding="utf-8"))
        return msgs, None
    if rep in CARD_DIRS:
        card_dir = vdir / CARD_DIRS[rep] / sid / f"level_{level}"
        cards = sorted(card_dir.glob("card*.png")) if card_dir.exists() else []
        if not cards:
            return None, f"missing cards at {card_dir}"
        md = vdir / "text" / sid / f"level_{level}.md"
        msgs = pl.build_messages(state, rep, card_paths=[str(c) for c in cards])
        return msgs, None
    return None, f"unknown representation {rep}"


def env_to_state(env, task, total_games):
    """Reset-walk to the task's game and replay the history actions.

    -> (replay_ok, note). Leaves the env positioned exactly at the decision
    step (the next env.step executes the branch action).
    """
    target = task["game_name"]
    acts = history_actions(task["state"])
    if len(acts) != task["state"]["step_idx"]:
        return False, (f"history has {len(acts)} actions but step_idx="
                       f"{task['state']['step_idx']}")
    for walk in range(total_games + 2):
        obs_raw, info = env.reset()
        got = game_name_of(info["extra.gamefile"][0])
        if got == target:
            break
    else:
        return False, f"game {target} not reached within one enumeration pass"

    for i, a in enumerate(acts):
        obs_list, _, dones, info = env.step([a])
        if dones[0]:
            return False, f"episode ended during history replay at step {i}"
    return True, None


def run_rollout(task, model, budget_slack, max_budget, max_gen_tokens):
    state = task["state"]
    rep = task["representation"]
    branch = task["branch"]

    messages, err = build_probe_messages(state, rep, task["variants_dir"], task["level"])
    if messages is None:
        return {"ok": False, "error": err}

    branch_action = (lex.normalize_action(state["correct_action"])
                     if branch == "correct"
                     else lex.normalize_action(state["distractor_action"]))

    env = task["_env"]
    total_games = task["_total_games"]
    ok, note = env_to_state(env, task, total_games)
    if not ok:
        return {"ok": False, "error": f"env replay: {note}"}

    remaining = max(1, state["total_steps"] - state["step_idx"])
    budget = min(max_budget, remaining + budget_slack)

    # ── the branch step itself ──
    obs_list, _, dones, info = env.step([branch_action])
    branch_feedback = process_ob(obs_list[0])
    branch_nothing = branch_feedback.strip().lower().startswith("nothing happens")

    messages.append({"role": "assistant",
                     "content": f"{pl.FORCED_THOUGHT}\nAction: {branch_action}"})
    messages.append({"role": "user", "content": f"Observation: {branch_feedback}"})

    actions, nothing_happened = [], 0
    success = bool(dones[0])
    terminated = "done_at_branch" if dones[0] else None
    steps = 0

    # ── continue with the model's own greedy policy ──
    while not dones[0] and steps < budget:
        text, _ = pl.generate_action(messages, model=model, max_tokens=max_gen_tokens)
        action = lex.parse_action_from_response(text)
        if not action:
            terminated = "unparsable_action"
            break
        actions.append(action)
        messages.append({"role": "assistant", "content": text})
        obs_list, _, dones, info = env.step([action])
        feedback = process_ob(obs_list[0])
        messages.append({"role": "user", "content": f"Observation: {feedback}"})
        steps += 1
        if feedback.strip().lower().startswith("nothing happens"):
            nothing_happened += 1
    if dones[0]:
        success = True
        terminated = "done"
    elif terminated is None:
        terminated = "budget_exhausted"

    return {
        "ok": True,
        "state_id": state["state_id"],
        "level": task["level"],
        "representation": rep,
        "branch": branch,
        "game_name": state["game_name"],
        "task_type": state["task_type"],
        "step_idx": state["step_idx"],
        "correct_action": lex.normalize_action(state["correct_action"]),
        "competitor_action": lex.normalize_action(state["distractor_action"]),
        "branch_action": branch_action,
        "branch_feedback": branch_feedback,
        "branch_nothing_happened": branch_nothing,
        "success": success,
        "terminated": terminated,
        "policy_steps_after_branch": steps,
        "budget": budget,
        "nothing_happened_steps": nothing_happened,
        "policy_actions": actions,
    }


# ══════════════════════════════════════════════════════════════════════
# Worker (owns one env)
# ══════════════════════════════════════════════════════════════════════

def _worker(task_queue, result_queue, ports, counter, args_dict):
    wid = os.getpid()
    pl.set_endpoint(ports[wid % len(ports)])
    env, _ = build_env(args_dict["config"], args_dict["split"])
    total_games = len(list(env.gamefiles))
    while True:
        task = task_queue.get()
        if task is None:
            break
        with counter.get_lock():
            order = counter.value
            counter.value += 1
        pl.set_endpoint(ports[order % len(ports)])
        task["_env"] = env
        task["_total_games"] = total_games
        try:
            result = run_rollout(task, args_dict["model"], args_dict["budget_slack"],
                                 args_dict["max_budget"], args_dict["max_gen_tokens"])
            if not result.get("ok") and "not reached within one enumeration" in str(
                    result.get("error")):
                # The worker-owned env's reset cycle desyncs after some history
                # replays (a fresh env enumerates all 134
                # games in one pass, a desynced one misses ~15% of states
                # forever). Rebuild the env and retry the task once.
                try:
                    env.close()
                except Exception:
                    pass
                env, _ = build_env(args_dict["config"], args_dict["split"])
                task["_env"] = env
                result = run_rollout(task, args_dict["model"], args_dict["budget_slack"],
                                     args_dict["max_budget"], args_dict["max_gen_tokens"])
            if result.get("ok"):
                out_file = Path(args_dict["output_dir"]) / task["out_name"]
                out_file.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                                    encoding="utf-8")
                result_queue.put((task["task_id"], True, task["out_name"]))
            else:
                result_queue.put((task["task_id"], False,
                                  f"{task['out_name']}: {result.get('error')}"))
        except Exception as exc:
            import traceback
            traceback.print_exc()
            result_queue.put((task["task_id"], False, f"{task['out_name']}: {exc}"))
    try:
        env.close()
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════
# Task enumeration
# ══════════════════════════════════════════════════════════════════════

def build_tasks(args):
    states_dir = Path(args.states_dir)
    out_dir = Path(args.output_dir)
    levels = [int(x) for x in args.levels.split(",")]
    reps = [r.strip() for r in args.representations.split(",") if r.strip()]
    branches = [b.strip() for b in args.branches.split(",") if b.strip()]
    for b in branches:
        if b not in ("correct", "competitor"):
            raise SystemExit(f"unknown branch '{b}'")

    manifest = json.loads((states_dir / "manifest.json").read_text(encoding="utf-8"))
    entries = manifest["states"]
    if args.max_states > 0:
        entries = entries[:args.max_states]

    tasks, missing = [], []
    tid = 0
    # Sorted by game so consecutive queue items share a game (the reset-walk
    # still pays per task, but logs stay readable and QA can group by game).
    entries = sorted(entries, key=lambda e: (e["game_name"], e["step_idx"]))
    for info in entries:
        state_id = info["state_id"]
        sf = states_dir / f"{state_id}.json"
        if not sf.exists():
            missing.append(f"state file {state_id}")
            continue
        state = json.loads(sf.read_text(encoding="utf-8"))
        if not state.get("competitor_admissible"):
            missing.append(f"{state_id}: competitor_admissible flag missing — "
                           f"wrong states dir?")
            continue
        for level in levels:
            for rep in reps:
                for branch in branches:
                    out_name = f"{state_id}__k{level}__{rep}__rollout_{branch}.json"
                    if (out_dir / out_name).exists() and not args.overwrite:
                        continue
                    tasks.append({
                        "task_id": tid,
                        "state": state,
                        "game_name": state["game_name"],
                        "level": level,
                        "representation": rep,
                        "branch": branch,
                        "variants_dir": args.variants_dir,
                        "out_name": out_name,
                    })
                    tid += 1
    return tasks, missing, manifest


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks, missing, manifest = build_tasks(args)
    print(f"[INFO] {manifest['total_states']} states; {len(tasks)} rollouts to run "
          f"(levels={args.levels}, reps={args.representations}, branches={args.branches})")
    if missing:
        print(f"[WARN] {len(missing)} inputs unusable, e.g. {missing[:3]}")
    if args.limit > 0:
        tasks = tasks[:args.limit]
        print(f"[INFO] --limit -> running only {len(tasks)} rollouts")
    if args.dry_run:
        print(f"[DRY RUN] {len(tasks)} rollouts, ~{len(tasks) * args.max_budget} "
              f"generations worst case. Nothing executed.")
        return
    if not tasks:
        print("Nothing to do — all rollouts already present.")
        return

    ports = [int(p.strip()) for p in args.ports.split(",")]
    workers = max(1, min(args.max_workers, len(tasks)))
    print(f"[INFO] {workers} env-owning workers over ports {ports}")

    args_dict = {
        "model": args.model, "config": args.config, "split": args.split,
        "budget_slack": args.budget_slack, "max_budget": args.max_budget,
        "max_gen_tokens": args.max_gen_tokens, "output_dir": str(out_dir),
    }

    ctx = multiprocessing.get_context("fork")
    task_queue, result_queue = ctx.Queue(), ctx.Queue()
    counter = ctx.Value("i", 0)
    for t in tasks:
        task_queue.put(t)
    for _ in range(workers):
        task_queue.put(None)

    procs = []
    for _ in range(workers):
        p = ctx.Process(target=_worker,
                        args=(task_queue, result_queue, ports, counter, args_dict),
                        daemon=True)
        p.start()
        procs.append(p)

    ok = fail = 0
    pbar = tqdm(total=len(tasks), desc="Rollouts")
    for _ in range(len(tasks)):
        try:
            _, success, note = result_queue.get(timeout=args.timeout)
        except queue.Empty:
            print(f"\n[WARN] no result for {args.timeout}s; stopping early.")
            break
        ok += int(success)
        fail += int(not success)
        if not success:
            print(f"[FAIL] {note}")
        pbar.update(1)
        pbar.set_postfix({"ok": ok, "fail": fail})
    pbar.close()

    for p in procs:
        p.join(timeout=60)
    for p in procs:
        if p.is_alive():
            p.terminate()

    print(f"\nDone. {ok} rollouts written, {fail} failed. Results in {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--states_dir", required=True)
    p.add_argument("--variants_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--config", default=str(CODE_DIR / "base_config.yaml"))
    p.add_argument("--split", default="eval_out_of_distribution")
    p.add_argument("--levels", default="0,8",
                   help="k levels to roll out (default: the two ends of the sweep)")
    p.add_argument("--representations", default="text",
                   help="rollout is expensive; default text only — add "
                        "structural/rasterized when the probe bridge is clear")
    p.add_argument("--branches", default="correct,competitor")
    p.add_argument("--budget_slack", type=int, default=20,
                   help="extra steps allowed beyond the expert's remaining steps")
    p.add_argument("--max_budget", type=int, default=40,
                   help="hard cap on policy steps after the branch action")
    p.add_argument("--max_gen_tokens", type=int, default=256)
    p.add_argument("--max_states", type=int, default=-1,
                   help="roll out only the first N states (cost control)")
    p.add_argument("--max_workers", type=int, default=4,
                   help="each worker owns one ALFWorld env — keep this small")
    p.add_argument("--ports", default="8000,8001,8002,8003")
    p.add_argument("--timeout", type=int, default=3600)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--limit", type=int, default=-1)
    p.add_argument("--dry_run", action="store_true")
    os.environ.setdefault("ALFWORLD_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "data", "alfworld"))
    main(p.parse_args())
