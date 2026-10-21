"""
Baseline (no-skill) WebShop evaluation via DashScope API.

Standard agent loop WITHOUT the SkillNet pipeline:
  - LLM directly generates Thought + Action at each step
  - No skill retrieval, no procedure generation, no code generation

Execution model: kimi-k2.5 via the DashScope API.
Dynamic scheduling — all games go into one shared queue and every worker
process pulls the next game from that queue as soon as it finishes.
"""
import os
import sys

os.environ['JAVA_HOME'] = os.path.join(
    os.path.dirname(os.path.abspath(sys.executable)),
    '..', 'lib', 'python3.10', 'site-packages', 'jdk4py', 'java-runtime'
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spacy_patch import patch_spacy
patch_spacy()

os.environ.setdefault('API_KEY', 'EMPTY')

import re
import json
import argparse
import contextlib
import traceback
import concurrent.futures
import multiprocessing
from pathlib import Path
from tqdm import tqdm
from retry import retry
from openai import OpenAI

SCRIPT_DIR = Path(__file__).resolve().parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

VIS_SKILL_ALL_ROOT = SCRIPT_DIR.parent.parent
WEBSHOP_ROOT = VIS_SKILL_ALL_ROOT  # experiments/webshop: shared web_agent_site/, data/, search_engine/
if str(WEBSHOP_ROOT) not in sys.path:
    sys.path.insert(0, str(WEBSHOP_ROOT))

SHARED_DATA_DIR = WEBSHOP_ROOT / "data"

from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
from prompts.system_prompt import webshop_system_prompt

DEFAULT_MODEL = "kimi-k2.5"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

_client = None
_TEMPERATURE = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "DASHSCOPE_API_KEY is not set. Export your own DashScope "
                "API key first, e.g.:  export DASHSCOPE_API_KEY=<your-key>"
            )
        base_url = os.environ.get("DASHSCOPE_BASE_URL", DEFAULT_BASE_URL)
        _client = OpenAI(api_key=api_key, base_url=base_url, timeout=300)
    return _client


def _sampling_kwargs():
    kwargs = {}
    if _TEMPERATURE is not None:
        kwargs["temperature"] = _TEMPERATURE
    return kwargs


@retry(tries=5, delay=5, backoff=2, jitter=(1, 3))
def llm(prompt, model=DEFAULT_MODEL):
    if isinstance(prompt, list):
        messages = prompt
    elif isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]
    else:
        raise ValueError(f"prompt must be list or str, got {type(prompt)}")
    response = _get_client().chat.completions.create(
        model=model,
        messages=messages,
        **_sampling_kwargs(),
    )
    content = response.choices[0].message.content
    return content if content is not None else "Output Error"


class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'


def parse_action(response: str) -> str:
    pattern = re.compile(r"Action:\s*(.+)", re.IGNORECASE)
    match = pattern.search(response)
    if match:
        return match.group(1).strip().strip('"\'*`')
    return ""


def run_standard_procedure(env, llm_func, model, messages, max_steps):
    task_done = False
    task_reward = 0
    current_steps = 0
    while not task_done and current_steps < max_steps:
        current_steps += 1
        try:
            response = llm_func(messages, model)
            print(f'{Colors.GREEN}Agent response: \n{response}{Colors.RESET}')
        except Exception as e:
            print(f'{Colors.RED}Error in LLM call: {e}{Colors.RESET}')
            break
        messages.append({"role": "assistant", "content": response})
        action = parse_action(response)
        observation, reward, done, info = env.step(action)
        task_reward = reward
        task_done = done
        print(f'{Colors.YELLOW}Observation: \n{observation}{Colors.RESET}')
        messages.append({"role": "user", "content": f"Observation: {observation}"})
        if task_done:
            print(f'{Colors.GREEN}Task completed! Reward: {task_reward}{Colors.RESET}')
    return messages, task_done, task_reward, current_steps


_env = None


def _init_worker(counter, data_path, init_sem=None):
    global _env, _client
    with counter.get_lock():
        worker_idx = counter.value
        counter.value += 1
    _client = None
    ctx = init_sem if init_sem is not None else contextlib.nullcontext()
    with ctx:
        print(f"[Worker {os.getpid()}] (idx={worker_idx}), loading WebShop env...")
        _env = WebAgentTextEnv(
            observation_mode='text',
            file_path=data_path,
            human_goals=True,
            num_products=None,
        )
        print(f"[Worker {os.getpid()}] WebShop env ready.")


def webshop_run_single(env, ob, instruction_text, max_steps=30, model=None):
    query = instruction_text
    messages = [{"role": "system", "content": webshop_system_prompt}]
    messages.append({"role": "user", "content": ob})

    messages, task_done, task_reward, steps = run_standard_procedure(
        env, llm, model, messages, max_steps
    )

    return {
        "query": query,
        "task_done": task_done,
        "reward": task_reward,
        "steps": steps,
        "messages": messages,
        "relevant_skill_names": [],
    }


def eval_single_game(game_idx, session_id, args, output_path):
    global _env
    try:
        ob = _env.reset(session=session_id)[0]
        instruction_text = _env.instruction_text
        print(f'{Colors.RED}[Game {game_idx}] Session {session_id}: {instruction_text}{Colors.RESET}')

        result = webshop_run_single(
            env=_env,
            ob=ob,
            instruction_text=instruction_text,
            max_steps=args.max_steps,
            model=args.model,
        )
        result['session_id'] = session_id

        save_file = os.path.join(output_path, f'idx_{game_idx}.json')
        with open(save_file, 'w') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        return result

    except Exception as e:
        print(f"Error in game {game_idx} (Session {session_id}):")
        traceback.print_exc()
        return None


def main(args):
    global _TEMPERATURE
    _TEMPERATURE = args.temperature

    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    model_name = args.model.split("/")[-1]
    output_path = os.path.join(str(SCRIPT_DIR), f'results/{model_name}/{args.exp_name}_skill_False')
    os.makedirs(output_path, exist_ok=True)

    test_indices_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_indices.json')
    with open(test_indices_path) as f:
        session_ids = json.load(f)

    num_games = len(session_ids)
    game_start = args.game_start if args.game_start >= 0 else 0
    game_end = args.game_end if args.game_end > 0 else num_games
    game_end = min(game_end, num_games)

    existing = set()
    finished_games = 0
    all_rewards = 0.0
    all_steps = 0
    if os.path.exists(output_path):
        for file in os.listdir(output_path):
            if file.endswith('.json') and file.startswith('idx_'):
                try:
                    idx = int(file.split('_')[1].split('.')[0])
                    existing.add(idx)
                    with open(os.path.join(output_path, file)) as f:
                        res = json.load(f)
                    all_rewards += float(res['reward'])
                    all_steps += int(res['steps'])
                    finished_games += 1
                except Exception:
                    continue

    tasks_to_run = {}
    for idx in range(game_start, game_end):
        if idx not in existing:
            tasks_to_run[idx] = session_ids[idx]

    print(f"Range [{game_start}, {game_end}), finished={finished_games}, remaining={len(tasks_to_run)}")
    if not tasks_to_run:
        if finished_games > 0:
            print(f"Avg Reward: {all_rewards / finished_games * 100:.2f}%, "
                  f"Avg Steps: {all_steps / finished_games:.2f}")
        return

    data_path = str(SHARED_DATA_DIR / 'items_shuffle.json')
    max_workers = args.max_workers

    print(f"Model: {args.model} via DashScope API "
          f"({os.environ.get('DASHSCOPE_BASE_URL', DEFAULT_BASE_URL)})")
    print(f"Dynamic scheduling: {max_workers} workers pulling from ONE shared queue ...")

    init_sem = multiprocessing.Semaphore(1)

    def make_executor(counter):
        return concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_init_worker,
            initargs=(counter, data_path, init_sem),
        )

    remaining = dict(tasks_to_run)
    crashes = 0
    MAX_CRASH_RESPAWNS = 5
    while remaining:
        worker_counter = multiprocessing.Value('i', 0)
        executor = make_executor(worker_counter)

        all_futures = {}
        for idx, sid in remaining.items():
            fut = executor.submit(eval_single_game, idx, sid, args, output_path)
            all_futures[fut] = idx

        pbar = tqdm(total=len(remaining), desc=f"WebShop-Baseline({model_name}, DashScope API)")
        pool_broken = False
        try:
            for future in concurrent.futures.as_completed(all_futures, timeout=600):
                idx = all_futures[future]
                try:
                    result = future.result()
                    if result:
                        finished_games += 1
                        all_rewards += float(result['reward'])
                        all_steps += int(result['steps'])
                        avg_r = all_rewards / finished_games
                        avg_s = all_steps / finished_games
                        pbar.set_postfix({'AvgR': f'{avg_r*100:.2f}%', 'Steps': f'{avg_s:.2f}'})
                except concurrent.futures.process.BrokenProcessPool:
                    print(f'\nGame {idx}: worker pool crashed; will respawn and retry unfinished tasks.')
                    pool_broken = True
                except Exception as exc:
                    print(f'\nGame {idx} exception: {exc}')
                pbar.update(1)
        except (TimeoutError, concurrent.futures.TimeoutError):
            print(f'\n[Scheduler] Deadlock detected (no progress for 600s). Respawning pool...')
            pool_broken = True
        pbar.close()
        executor.shutdown(wait=False, cancel_futures=True)

        if not pool_broken:
            break

        crashes += 1
        if crashes > MAX_CRASH_RESPAWNS:
            print(f'[Scheduler] {MAX_CRASH_RESPAWNS} pool crashes; aborting.')
            break
        finished_games = 0
        all_rewards = 0.0
        all_steps = 0
        done = set()
        for file in os.listdir(output_path):
            if file.endswith('.json') and file.startswith('idx_'):
                try:
                    idx = int(file.split('_')[1].split('.')[0])
                    done.add(idx)
                    with open(os.path.join(output_path, file)) as f:
                        res = json.load(f)
                    all_rewards += float(res['reward'])
                    all_steps += int(res['steps'])
                    finished_games += 1
                except Exception:
                    continue
        remaining = {idx: sid for idx, sid in tasks_to_run.items() if idx not in done}
        print(f'[Scheduler] respawn #{crashes}: {len(remaining)} tasks left')

    print(f"\nDone. Finished: {finished_games}/{num_games}")
    if finished_games > 0:
        print(f"Avg Reward: {all_rewards / finished_games * 100:.2f}%")
        print(f"Avg Steps: {all_steps / finished_games:.2f}")

    sr_count = 0
    for file in os.listdir(output_path):
        if file.endswith('.json') and file.startswith('idx_'):
            try:
                with open(os.path.join(output_path, file)) as f:
                    res = json.load(f)
                if float(res['reward']) == 1.0:
                    sr_count += 1
            except Exception:
                continue
    summary = {
        "total_games": finished_games,
        "score": f"{all_rewards / finished_games * 100:.2f}" if finished_games > 0 else "0.00",
        "sr_pct": f"{sr_count / finished_games * 100:.2f}" if finished_games > 0 else "0.00",
        "avg_steps": f"{all_steps / finished_games:.2f}" if finished_games > 0 else "0.00",
        "model": args.model,
        "max_steps": args.max_steps,
        "use_skill": False,
        "experiment": "baseline-noskill",
    }
    with open(os.path.join(output_path, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--max_workers', type=int, default=20,
                        help="number of dynamic worker processes (keep moderate "
                             "to respect DashScope API rate limits)")
    parser.add_argument('--max_steps', type=int, default=30)
    parser.add_argument('--exp_name', default='webshop_baseline')
    parser.add_argument('--temperature', type=float, default=None,
                        help="sampling temperature (default: not passed, API default)")
    parser.add_argument('--game_start', type=int, default=-1)
    parser.add_argument('--game_end', type=int, default=-1)
    args = parser.parse_args()
    main(args)
