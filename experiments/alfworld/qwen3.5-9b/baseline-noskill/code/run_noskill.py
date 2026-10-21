"""
Baseline experiment — NO SKILL.

Direct task execution with Qwen3.5-9B, no skill retrieval or procedure generation.
The agent receives only the system prompt and task observation, then acts step by step.
"""
import os
import re
import json
import yaml
import argparse
import concurrent.futures
from pathlib import Path
from tqdm import tqdm
from retry import retry
from openai import OpenAI

os.environ.setdefault("ALFWORLD_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "..", "data", "alfworld"))

import alfworld
import alfworld.agents.environment
from alfworld.agents.environment import get_environment

# ── System prompt (identical to SkillNet) ──

alfworld_system_prompt = """Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish. 
For each of your turn, you will be given the observation of the last turn. You should first think about the current condition and plan for your future actions, and then output your action in this turn. Your output must strictly follow this format:"Thought: your thoughts.\nAction: your next action".

The available actions are:
1. go to {recep}
2. take {obj} from {recep}
3. move {obj} to {recep}
4. open {recep}
5. close {recep}
6. use {obj}
7. clean {obj} with {recep}
8. heat {obj} with {recep}
9. cool {obj} with {recep}
where {obj} and {recep} correspond to objects and receptacles.
After your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the envrionment output "Nothing happened", that means the previous action is invalid and you should try more options.

Your response should use the following format:

Thought: <your thoughts>
Action: <your next action>"""


# ── Globals set per-process ──
_client = None
_port = 8000

DEFAULT_MODEL = os.path.join(os.environ.get("MODELS_DIR", os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "..", "models"))), "Qwen3.5-9B")


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(
            api_key="EMPTY",
            base_url=f"http://localhost:{_port}/v1",
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


def process_ob(ob):
    if ob.startswith("You arrive at loc "):
        ob = ob[ob.find(". ") + 2:]
    return ob


def parse_action(response: str) -> str:
    m = re.compile(r"Action:\s*(.+)", re.IGNORECASE).search(response)
    if m:
        return m.group(1).strip().strip("\"'*`")
    return ""


def run_standard_procedure(env, llm_fn, model, process_ob_fn, messages, max_steps):
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
        observation, task_reward, done, info = env.step([action])
        observation, task_reward, task_done = process_ob_fn(observation[0]), info["won"][0], done[0]
        messages.append({"role": "user", "content": f"Observation: {observation}"})
        if task_done:
            print(f"{Colors.GREEN}Task completed! Reward: {task_reward}{Colors.RESET}")
    return messages, task_done, task_reward, current_steps


# ── Core execution ──

def alfworld_run_single(env, obs=[], names=[], max_steps=30, model=None):
    results = []
    for task_idx, (ob, name) in enumerate(zip(obs, names)):
        print(f"{Colors.RED}Processing task {task_idx+1}/{len(obs)}: {name}{Colors.RESET}")
        messages = [{"role": "system", "content": alfworld_system_prompt}]
        messages.append({"role": "user", "content": ob})

        messages, task_done, task_reward, steps = run_standard_procedure(
            env, llm, model, process_ob, messages, max_steps
        )

        query = ob.split("Your task is to: ")[-1].split("\n")[0].strip()
        results.append({
            "query": query,
            "name": name,
            "task_done": task_done,
            "reward": task_reward,
            "steps": steps,
            "messages": messages,
        })
    return results


def _init_worker(port):
    global _port, _client
    _port = port
    _client = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def eval_single_game(game_idx, args, config, split, output_path):
    env = None
    try:
        env = get_environment(config["env"]["type"])(config, train_eval=split)
        env = env.init_env(batch_size=1)
        for _ in range(game_idx + 1):
            obs_list, info = env.reset()

        ob_str = "\n".join(obs_list[0].split("\n\n")[1:])
        game_name = "/".join(info["extra.gamefile"][0].split("/")[-3:-1])

        batch_results = alfworld_run_single(
            env=env,
            obs=[ob_str],
            names=[game_name],
            max_steps=args.max_steps,
            model=args.model,
        )
        result = batch_results[0]

        save_file = f"{output_path}/idx_{game_idx}.json"
        with open(save_file, "w") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        return result

    except Exception as e:
        print(f"Error in game {game_idx}: {e}")
        import traceback; traceback.print_exc()
        return None
    finally:
        if env:
            env.close()


# ── Summary ──

def generate_summary(output_path, summary_path):
    results = []
    for f in sorted(Path(output_path).glob("idx_*.json"), key=lambda p: int(p.stem.split("_")[1])):
        with open(f) as fh:
            results.append(json.load(fh))

    if not results:
        print("[WARN] No results to summarize.")
        return

    def classify_task(name):
        name_lower = name.lower()
        if "pick_two" in name_lower: return "pick_two"
        if "pick_and_place" in name_lower: return "pick_place"
        if "pick_clean" in name_lower: return "pick_clean"
        if "pick_heat" in name_lower: return "pick_heat"
        if "pick_cool" in name_lower: return "pick_cool"
        if "look_at" in name_lower or "examine" in name_lower: return "examine"
        return "other"

    total = len(results)
    success = 0
    total_steps = 0
    by_type = {}

    for r in results:
        reward = r.get("reward", 0)
        if isinstance(reward, list):
            reward = reward[-1] if reward else 0
        won = reward in [True, 1]
        steps = r.get("steps", 0)
        task_type = classify_task(r.get("name", ""))

        if won: success += 1
        total_steps += steps

        if task_type not in by_type:
            by_type[task_type] = {"total": 0, "success": 0, "steps": 0}
        by_type[task_type]["total"] += 1
        if won: by_type[task_type]["success"] += 1
        by_type[task_type]["steps"] += steps

    summary = {
        "experiment": "baseline-noskill-9B",
        "description": "No-skill baseline with Qwen3.5-9B, direct task execution",
        "model": "Qwen3.5-9B",
        "benchmark": "ALFWorld (eval_out_of_distribution)",
        "overall": {
            "total": total,
            "success": success,
            "success_rate": round(success / total * 100, 2) if total else 0,
            "avg_steps": round(total_steps / total, 1) if total else 0,
        },
        "by_task_type": {},
    }

    for t in ["pick_place", "pick_clean", "pick_heat", "pick_cool", "pick_two", "examine"]:
        if t in by_type:
            d = by_type[t]
            summary["by_task_type"][t] = {
                "total": d["total"],
                "success": d["success"],
                "success_rate": round(d["success"] / d["total"] * 100, 1) if d["total"] else 0,
                "avg_steps": round(d["steps"] / d["total"], 1) if d["total"] else 0,
            }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"SUMMARY: {summary['experiment']}")
    print(f"{'='*60}")
    print(f"Overall: {success}/{total} = {summary['overall']['success_rate']}% SR, "
          f"avg {summary['overall']['avg_steps']} steps")
    print(f"\nPer task type:")
    for t, d in summary["by_task_type"].items():
        print(f"  {t:<12} {d['success']:>2}/{d['total']:<2} = {d['success_rate']:>5.1f}%  "
              f"avg {d['avg_steps']:.1f} steps")
    print(f"\nSaved to {summary_path}")


# ── Main ──

def main(args):
    config_path = Path(__file__).resolve().parent / "base_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    split = "eval_out_of_distribution"

    model_name = args.model.split("/")[-1]
    output_path = f"{PROJECT_ROOT}/results/alfworld/{model_name}/noskill_baseline"
    os.makedirs(output_path, exist_ok=True)

    temp_env = get_environment(config["env"]["type"])(config, train_eval=split)
    temp_env = temp_env.init_env(batch_size=1)
    total_games = len(temp_env.gamefiles)
    del temp_env

    game_start = args.game_start if args.game_start >= 0 else 0
    game_end = args.game_end if args.game_end > 0 else total_games
    game_end = min(game_end, total_games)

    ports = [int(p.strip()) for p in args.ports.split(",")]
    print(f"Total games: {total_games}, running [{game_start}, {game_end}), ports={ports}")

    existing = set()
    finished_games = 0
    all_rewards = 0
    all_steps = 0
    if os.path.exists(output_path):
        for file in os.listdir(output_path):
            if file.endswith(".json") and file.startswith("idx_"):
                try:
                    idx = int(file.split("_")[1].split(".")[0])
                    existing.add(idx)
                    with open(f"{output_path}/{file}") as f:
                        res = json.load(f)
                    r = res["reward"]
                    if isinstance(r, list):
                        r = r[-1] if r else 0
                    if r in [True, 1]:
                        all_rewards += 1
                    all_steps += res["steps"]
                    finished_games += 1
                except:
                    continue

    tasks_to_run = [i for i in range(game_start, game_end) if i not in existing]
    print(f"Already finished: {finished_games}, Remaining: {len(tasks_to_run)}")

    if tasks_to_run:
        max_workers = args.max_workers
        workers_per_port = max_workers // len(ports)
        print(f"Starting with {max_workers} workers across ports {ports} ({workers_per_port}/port)...")

        all_futures = {}
        executors = []
        for pi, port in enumerate(ports):
            n_workers = workers_per_port if pi < len(ports) - 1 else max_workers - workers_per_port * (len(ports) - 1)
            ex = concurrent.futures.ProcessPoolExecutor(
                max_workers=n_workers,
                initializer=_init_worker,
                initargs=(port,),
            )
            executors.append(ex)

        for i, idx in enumerate(tasks_to_run):
            ex = executors[i % len(executors)]
            fut = ex.submit(eval_single_game, idx, args, config, split, output_path)
            all_futures[fut] = idx

        pbar = tqdm(total=len(tasks_to_run), desc=f"Eval(ports={ports})")
        for future in concurrent.futures.as_completed(all_futures):
            idx = all_futures[future]
            try:
                result = future.result()
                if result:
                    finished_games += 1
                    r = result["reward"]
                    if isinstance(r, list):
                        r = r[-1] if r else 0
                    if r in [True, 1]:
                        all_rewards += 1
                    all_steps += result["steps"]
                    sr = all_rewards / finished_games if finished_games else 0
                    pbar.set_postfix({"SR": f"{sr:.2f}", "Steps": f"{all_steps/finished_games:.1f}"})
            except Exception as exc:
                print(f"\nGame {idx} exception: {exc}")
            pbar.update(1)
        pbar.close()

        for ex in executors:
            ex.shutdown(wait=False)

    print(f"\nDone. Finished: {finished_games}")
    if finished_games > 0:
        print(f"Avg Reward: {all_rewards/finished_games:.4f}")
        print(f"Avg Steps: {all_steps/finished_games:.1f}")

    all_done = all(
        (Path(output_path) / f"idx_{i}.json").exists()
        for i in range(game_start, game_end)
    )
    if all_done:
        summary_path = f"{output_path}/summary_noskill_baseline.json"
        generate_summary(output_path, summary_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max_workers", type=int, default=32)
    parser.add_argument("--max_steps", type=int, default=30)
    parser.add_argument("--ports", type=str, default="8000,8001,8002,8003")
    parser.add_argument("--game_start", type=int, default=-1)
    parser.add_argument("--game_end", type=int, default=-1)
    args = parser.parse_args()
    main(args)
