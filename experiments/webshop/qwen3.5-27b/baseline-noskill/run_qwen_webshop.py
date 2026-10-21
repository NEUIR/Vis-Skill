"""
Baseline (no-skill) WebShop evaluation with Qwen3.5-27B via vLLM.

Standard agent loop WITHOUT SkillNet pipeline:
  - LLM directly generates Thought + Action at each step
  - No skill retrieval, no procedure generation, no code generation

This is the baseline to compare against SkillNet's 3-stage pipeline.

Paths:
  - Backbone:       models/Qwen3.5-27B
  - Dataset:        ../../data/items_shuffle.json  (shared experiments/webshop/data)
  - web_agent_site: ../../web_agent_site           (shared; resolves ../../search_engine
                    and ../../data via its own relative paths)
  - Output:         ./results/
"""
import os
import sys

os.environ['JAVA_HOME'] = os.path.join(
    os.path.dirname(os.path.abspath(sys.executable)),
    '..', 'lib', 'python3.10', 'site-packages', 'jdk4py', 'java-runtime'
)

SCRIPT_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
WEBSHOP_ROOT_PATH = os.path.dirname(os.path.dirname(SCRIPT_DIR_PATH))  # experiments/webshop/
sys.path.insert(0, SCRIPT_DIR_PATH)    # experiment-local: prompts/, spacy_patch.py
sys.path.insert(0, WEBSHOP_ROOT_PATH)  # shared copy: web_agent_site/
from spacy_patch import patch_spacy
patch_spacy()

os.environ.setdefault('API_KEY', 'EMPTY')
os.environ.setdefault('BASE_URL', 'http://localhost:8000/v1')

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

SCRIPT_DIR = Path(__file__).resolve().parent
WEBSHOP_ROOT = SCRIPT_DIR.parent.parent  # experiments/webshop/ — shared data/ & web_agent_site/

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
from prompts.system_prompt import webshop_system_prompt

DEFAULT_MODEL = os.path.join(os.environ.get("MODELS_DIR", os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "models"))), "Qwen3.5-27B")

_client = OpenAI(
    api_key=os.environ.get('API_KEY', 'EMPTY'),
    base_url=os.environ.get('BASE_URL', 'http://localhost:8000/v1'),
)


@retry(tries=5, delay=5, backoff=2, jitter=(1, 3))
def llm(prompt, model=DEFAULT_MODEL):
    if isinstance(prompt, list):
        messages = prompt
    elif isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]
    else:
        raise ValueError(f"prompt must be list or str, got {type(prompt)}")
    response = _client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
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
_port = None


def _init_worker(port, data_path):
    global _env, _client, _port
    _port = port
    _client = OpenAI(
        api_key=os.environ.get('API_KEY', 'EMPTY'),
        base_url=f'http://localhost:{port}/v1',
    )
    print(f"[Worker {os.getpid()}] port={port}, loading WebShop env...")
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
    from collections import deque

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
                except:
                    continue

    tasks_to_run = {}
    for idx in range(num_games):
        if idx not in existing:
            tasks_to_run[idx] = session_ids[idx]

    print(f"Total={num_games}, finished={finished_games}, remaining={len(tasks_to_run)}")
    if not tasks_to_run:
        if finished_games > 0:
            print(f"Avg Reward: {all_rewards / finished_games * 100:.2f}%, Avg Steps: {all_steps / finished_games:.2f}")
        return

    data_path = os.path.join(str(WEBSHOP_ROOT), 'data', 'items_shuffle.json')

    ports = [int(p.strip()) for p in args.ports.split(",")]
    max_workers = args.max_workers
    active_ports = ports[:max_workers] if max_workers < len(ports) else ports
    workers_per_port = max(1, max_workers // len(active_ports))
    print(f"Starting with {max_workers} workers across ports {active_ports} ({workers_per_port}/port)...")

    executors = []
    worker_counts = []
    for pi, port in enumerate(active_ports):
        n_workers = workers_per_port if pi < len(active_ports) - 1 else max_workers - workers_per_port * (len(active_ports) - 1)
        n_workers = max(1, n_workers)
        ex = concurrent.futures.ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_worker,
            initargs=(port, data_path),
        )
        executors.append(ex)
        worker_counts.append(n_workers)

    pending = deque(tasks_to_run.items())
    in_flight = {}

    def submit_next(ex):
        if pending:
            idx, sid = pending.popleft()
            fut = ex.submit(eval_single_game, idx, sid, args, output_path)
            in_flight[fut] = (idx, ex)

    for ex, n in zip(executors, worker_counts):
        for _ in range(n):
            submit_next(ex)

    pbar = tqdm(total=len(tasks_to_run), desc=f"WebShop-Baseline(ports={ports})")
    while in_flight:
        done, _ = concurrent.futures.wait(
            list(in_flight), return_when=concurrent.futures.FIRST_COMPLETED
        )
        for future in done:
            idx, ex = in_flight.pop(future)
            try:
                result = future.result()
                if result:
                    finished_games += 1
                    all_rewards += float(result['reward'])
                    all_steps += int(result['steps'])
                    avg_r = all_rewards / finished_games
                    avg_s = all_steps / finished_games
                    pbar.set_postfix({'AvgR': f'{avg_r*100:.2f}%', 'Steps': f'{avg_s:.2f}'})
            except Exception as exc:
                print(f'\nGame {idx} exception: {exc}')
            submit_next(ex)
            pbar.update(1)
    pbar.close()

    for ex in executors:
        ex.shutdown(wait=False)

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
            except:
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
    parser.add_argument('--max_workers', type=int, default=12)
    parser.add_argument('--max_steps', type=int, default=30)
    parser.add_argument('--exp_name', default='qwen_webshop_baseline')
    parser.add_argument('--ports', default='8000,8001,8002,8003')
    args = parser.parse_args()
    main(args)
