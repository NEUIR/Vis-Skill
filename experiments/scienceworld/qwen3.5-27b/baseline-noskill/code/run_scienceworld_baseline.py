"""
ScienceWorld baseline — NO SKILL (backbone: Qwen3.5-27B).

The agent receives only the system prompt + task description, then acts step
by step (Thought/Action loop). Tests the model's base capability without any
skill.

Protocol follows the official SkillNet ScienceWorld evaluation
(https://github.com/zjunlp/SkillNet):
  - test split: test_indices.json from SkillNet
    (211 episodes over 24 tasks, copied to this directory)
  - env: scienceworld.ScienceWorldEnv, simplificationStr="easy",
    envStepLimit = max_steps, score = running max of info['score'] (0-100)

Scheduling is fully DYNAMIC (two levels):
  1. Client side: all episodes are submitted to ONE shared
     ProcessPoolExecutor queue; every worker process, once it finishes an
     episode, pulls the next pending episode from that common queue.
  2. Server side: all requests go to a single online vLLM endpoint started
     with --data-parallel-size 4 (one DP rank per GPU); vLLM routes each
     request to the currently least-loaded rank, so every GPU stays busy
     until the very last request.
"""
import os
import re
import json
import argparse
import traceback
import concurrent.futures
import multiprocessing
from pathlib import Path
from tqdm import tqdm
from retry import retry
from openai import OpenAI
from scienceworld import ScienceWorldEnv

# ── System prompt (verbatim from SkillNet src/scienceworld/prompts/system_prompt.py) ──

scienceworld_system_prompt = """ You are a helpful assistant to do some scientific experiment in an environment.
In the environment, there are several rooms: kitchen, foundry, workshop, bathroom, outside, living room, bedroom, greenhouse, art studio, hallway
You should explore the environment and find the items you need to complete the experiment.
You can teleport to any room in one step.
All containers in the environment have already been opened, you can directly get items from the containers.
The available actions are:
    open OBJ: open a container
    close OBJ: close a container
    activate OBJ: activate a device
    deactivate OBJ: deactivate a device
    connect OBJ to OBJ: connect electrical components
    disconnect OBJ: disconnect electrical components
    use OBJ [on OBJ]: use a device/item
    look around: describe the current room
    examine OBJ: describe an object in detail
    look at OBJ: describe a container's contents
    read OBJ: read a note or book
    move OBJ to OBJ: move an object to a container
    pick up OBJ: move an object to the inventory
    pour OBJ into OBJ: pour a liquid into a container
    mix OBJ: chemically mix a container
    teleport to LOC: teleport to a specific room
    focus on OBJ: signal intent on a task object
    wait: task no action for 10 steps
    wait1: task no action for a step"

Your response should use the following format:

Thought: <your thoughts>
Action: <your next action>
"""

# ── Globals set per-process ──

_client = None
_port = 8000
_NUM_PORTS = 2

DEFAULT_MODEL = os.path.join(os.environ.get("MODELS_DIR", os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "..", "models"))), "Qwen3.5-27B")

CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[1]   # baseline-noskill/
INDICES_PATH = CODE_DIR / "test_indices.json"

_indices_cache = None


def _indices():
    global _indices_cache
    if _indices_cache is None:
        with open(INDICES_PATH) as f:
            _indices_cache = json.load(f)
    return _indices_cache


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(
            api_key="EMPTY",
            base_url=f"http://localhost:{_port}/v1",
            timeout=60.0,
        )
    return _client


@retry(tries=5, delay=5, backoff=2, jitter=(1, 3))
def llm(prompt, model=DEFAULT_MODEL):
    if isinstance(prompt, list):
        messages = prompt
    elif isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]
    else:
        raise ValueError(f"prompt must be a list or a string, got {type(prompt)}")
    response = _get_client().chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    content = response.choices[0].message.content
    return content if content is not None else "Output Error"


# ── Helpers ──

class Colors:
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"


def parse_action(response: str) -> str:
    m = re.compile(r"Action:\s*(.+)", re.IGNORECASE).search(response)
    if m:
        return m.group(1).strip().strip("\"'*`")
    return ""


def run_standard_procedure(env, llm_fn, model, messages, max_steps):
    task_done = False
    task_reward = 0
    current_steps = 0
    while not task_done and current_steps < max_steps:
        current_steps += 1
        try:
            response = llm_fn(messages, model)
        except Exception as e:
            print(f"{Colors.RED}LLM error: {e}{Colors.RESET}")
            break
        messages.append({"role": "assistant", "content": response})
        action = parse_action(response)
        observation, step_reward, task_done, info = env.step(action)
        if info.get("score") is not None and info["score"] > task_reward:
            task_reward = info["score"]
        messages.append({"role": "user", "content": f"Observation: {observation}"})
        if task_done:
            print(f"{Colors.GREEN}Task completed! Score: {task_reward}{Colors.RESET}")
    return messages, task_done, task_reward, current_steps


# ── Core execution ──

def scienceworld_run_single(env, task_name, var_idx, max_steps, model):
    obs, info = env.reset()
    query = env.get_task_description()

    print(f"{Colors.RED}Processing: {task_name} (Var: {var_idx}){Colors.RESET}")
    print(f"Query: {query}")

    messages = [{"role": "system", "content": scienceworld_system_prompt}]
    messages.append({"role": "user", "content": query})

    messages, task_done, task_reward, steps = run_standard_procedure(
        env, llm, model, messages, max_steps
    )

    return {
        "query": query,
        "name": task_name,
        "variation_idx": var_idx,
        "task_done": task_done,
        "reward": task_reward,
        "steps": steps,
        "messages": messages,
        "relevant_skill_names": [],
    }


def _init_worker(base_port):
    global _port, _client
    _port = base_port + (os.getpid() % _NUM_PORTS)
    _client = None


def eval_single_episode(idx, args, output_path):
    task_name, var_idx = _indices()[idx]
    env = None
    try:
        env = ScienceWorldEnv(taskName=None, serverPath=None, envStepLimit=args.max_steps)
        env.load(task_name, var_idx, simplificationStr="easy")

        result = scienceworld_run_single(env, task_name, var_idx, args.max_steps, args.model)

        save_file = os.path.join(output_path, f"idx_{idx}.json")
        with open(save_file, "w") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        return result

    except Exception as e:
        print(f"Error in episode {idx} ({task_name} var {var_idx}): {e}")
        traceback.print_exc()
        return None
    finally:
        if env:
            try:
                env.close()
            except Exception:
                pass


# ── Summary ──

def generate_summary(output_path, summary_path, model, max_steps):
    results = []
    for f in sorted(Path(output_path).glob("idx_*.json"), key=lambda p: int(p.stem.split("_")[1])):
        with open(f) as fh:
            results.append(json.load(fh))

    if not results:
        print("[WARN] No results to summarize.")
        return

    total = len(results)
    score_sum = 0.0
    steps_sum = 0
    done_count = 0
    by_task = {}

    for r in results:
        score = float(r.get("reward", 0) or 0)
        steps = int(r.get("steps", 0) or 0)
        done = bool(r.get("task_done", False))
        task = r.get("name", "unknown")

        score_sum += score
        steps_sum += steps
        if done:
            done_count += 1

        d = by_task.setdefault(task, {"count": 0, "score_sum": 0.0, "steps_sum": 0, "done": 0})
        d["count"] += 1
        d["score_sum"] += score
        d["steps_sum"] += steps
        if done:
            d["done"] += 1

    summary = {
        "experiment": "baseline-noskill-27B-scienceworld",
        "description": "No-skill baseline with Qwen3.5-27B on ScienceWorld, direct Thought/Action execution",
        "model": model,
        "benchmark": "ScienceWorld (SkillNet test split: 211 episodes, 24 tasks)",
        "max_steps": max_steps,
        "overall": {
            "total": total,
            "avg_score": round(score_sum / total, 2) if total else 0,
            "avg_steps": round(steps_sum / total, 1) if total else 0,
            "task_done": done_count,
            "task_done_rate": round(done_count / total * 100, 2) if total else 0,
        },
        "by_task": {},
    }

    for t in sorted(by_task.keys()):
        d = by_task[t]
        summary["by_task"][t] = {
            "count": d["count"],
            "avg_score": round(d["score_sum"] / d["count"], 2) if d["count"] else 0,
            "avg_steps": round(d["steps_sum"] / d["count"], 1) if d["count"] else 0,
            "task_done": d["done"],
        }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"SUMMARY: {summary['experiment']}")
    print(f"{'='*60}")
    print(f"Overall: avg score {summary['overall']['avg_score']}, "
          f"avg {summary['overall']['avg_steps']} steps, "
          f"done {done_count}/{total} ({summary['overall']['task_done_rate']}%)")
    print(f"\nPer task:")
    for t, d in summary["by_task"].items():
        print(f"  {t:<55} n={d['count']:>2}  score={d['avg_score']:>6.2f}  "
              f"steps={d['avg_steps']:>5.1f}  done={d['task_done']}/{d['count']}")
    print(f"\nSaved to {summary_path}")


# ── Main ──

def main(args):
    indices = _indices()
    num_episodes = len(indices)

    model_name = args.model.rstrip("/").split("/")[-1]
    output_path = f"{PROJECT_ROOT}/results/scienceworld/{model_name}/noskill_baseline"
    os.makedirs(output_path, exist_ok=True)

    ep_start = args.ep_start if args.ep_start >= 0 else 0
    ep_end = args.ep_end if args.ep_end > 0 else num_episodes
    ep_end = min(ep_end, num_episodes)

    existing = set()
    finished = 0
    all_rewards = 0.0
    all_steps = 0
    if os.path.exists(output_path):
        for file in os.listdir(output_path):
            if file.endswith(".json") and file.startswith("idx_"):
                try:
                    idx = int(file.split("_")[1].split(".")[0])
                    existing.add(idx)
                    with open(os.path.join(output_path, file)) as f:
                        res = json.load(f)
                    all_rewards += float(res.get("reward", 0) or 0)
                    all_steps += int(res.get("steps", 0) or 0)
                    finished += 1
                except Exception:
                    continue

    tasks_to_run = [i for i in range(ep_start, ep_end) if i not in existing]
    print(f"Total episodes: {num_episodes}, range [{ep_start}, {ep_end}), "
          f"finished={finished}, remaining={len(tasks_to_run)}, port={args.port}")

    if tasks_to_run:
        print(f"Starting with {args.max_workers} workers on a SHARED queue "
              f"(dynamic scheduling, endpoint port={args.port}) ...")

        # ONE executor = ONE common queue. Every worker pulls the next
        # episode as soon as it finishes its current one.
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.max_workers,
            initializer=_init_worker,
            initargs=(args.port,),
        ) as executor:
            future_to_idx = {
                executor.submit(eval_single_episode, idx, args, output_path): idx
                for idx in tasks_to_run
            }

            pbar = tqdm(total=len(tasks_to_run), desc=f"ScienceWorld-noskill(port={args.port})")
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    if result:
                        finished += 1
                        all_rewards += float(result.get("reward", 0) or 0)
                        all_steps += int(result.get("steps", 0) or 0)
                        pbar.set_postfix({
                            "AvgScore": f"{all_rewards / finished:.2f}",
                            "Steps": f"{all_steps / finished:.1f}",
                        })
                except Exception as exc:
                    print(f"\nEpisode {idx} exception: {exc}")
                pbar.update(1)
            pbar.close()

    print(f"\nDone. Finished: {finished}/{ep_end - ep_start}")
    if finished > 0:
        print(f"Avg Score: {all_rewards / finished:.2f}")
        print(f"Avg Steps: {all_steps / finished:.1f}")

    all_done = all(
        (Path(output_path) / f"idx_{i}.json").exists()
        for i in range(ep_start, ep_end)
    )
    if all_done:
        summary_path = os.path.join(output_path, "summary.json")
        generate_summary(output_path, summary_path, args.model, args.max_steps)


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max_workers", type=int, default=32)
    parser.add_argument("--max_steps", type=int, default=30)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--ep_start", type=int, default=-1)
    parser.add_argument("--ep_end", type=int, default=-1)
    args = parser.parse_args()
    main(args)
