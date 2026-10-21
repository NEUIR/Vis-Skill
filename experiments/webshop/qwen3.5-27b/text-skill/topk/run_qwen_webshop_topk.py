"""
SkillNet WebShop text-skill TOP-K evaluation — v1 (ASIN click fix).
TOP-K variant of text-skill: identical pipeline and prompts; the only
difference is how many skills are sliced from the annotation ranking
(--top_k; level0 baseline used 5: level-2->3, level+2->7, level+4->9,
level+8->13, level+16->21).

Reuses the fixed v1 assets from the parent directory (text-skill/):
prompts/system_prompt.py, prompt_generator.py, skills/, annotations,
test_indices. Shared dataset (experiments/webshop/data/) is read-only.
Results are written to text-skill/topk/results/.

Fixes vs. the pre-v1 controlled run — ALL PROMPT-LEVEL, dataset files
(data/webshop_overall_procedure_examples.txt, webshop_procedure_code_template.py,
items_shuffle.json, annotations, skills) are used UNCHANGED from the shared copies:
  - System prompt explicitly instructs ASIN-based product clicks
  - Stage-2 procedure-generation prompt adds instruction #5 (ASIN click format)
    plus an explicit correction of the few-shot examples' click-the-title format
  - Stage-3 code-generation prompt instructs invalid-action (stuck) detection
    with an ASIN-format hint to be implemented inside the generated code

Pipeline (2-stage, controlled):
  1. [SKIP] Skill retrieval — replaced by pre-annotated top-5 from annotations JSON
  2. LLM generates phased procedural guidance from skill contents
  3. LLM generates Python code -> exec() runs agent loop
"""
import os
import sys

os.environ['JAVA_HOME'] = os.path.join(
    os.path.dirname(os.path.abspath(sys.executable)),
    '..', 'lib', 'python3.10', 'site-packages', 'jdk4py', 'java-runtime'
)

SCRIPT_DIR_PATH = os.path.dirname(os.path.abspath(__file__))      # text-skill/topk
PARENT_DIR_PATH = os.path.dirname(SCRIPT_DIR_PATH)                # text-skill/ (fixed v1 prompts & helpers)
WEBSHOP_ROOT_PATH = os.path.dirname(os.path.dirname(PARENT_DIR_PATH))  # experiments/webshop/ (shared web_agent_site & data)
sys.path.insert(0, PARENT_DIR_PATH)
sys.path.insert(0, WEBSHOP_ROOT_PATH)
from spacy_patch import patch_spacy
patch_spacy()

os.environ.setdefault('API_KEY', 'EMPTY')
os.environ.setdefault('BASE_URL', 'http://localhost:8000/v1')

_port = 8000

import re
import json
import yaml
import argparse
import contextlib
import traceback
import fcntl
import concurrent.futures
import multiprocessing
from collections import deque
from pathlib import Path
from tqdm import tqdm
from retry import retry
from openai import OpenAI

SCRIPT_DIR = Path(__file__).resolve().parent   # text-skill/topk — results live here
PARENT = SCRIPT_DIR.parent                     # text-skill/ — fixed prompts, skills, annotations (symlinks)
WEBSHOP_ROOT = PARENT.parent.parent            # experiments/webshop/ — shared web_agent_site/ & data/ (read-only)

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
from prompts.system_prompt import webshop_system_prompt
from prompt_generator import (
    generate_overall_procedure_prompt,
    generate_overall_procedure_code_prompt,
)

DEFAULT_MODEL = os.path.join(os.environ.get("MODELS_DIR", os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "..", "models"))), "Qwen3.5-27B")

_client = None


def _sanitize_procedure_code(code: str) -> str:
    """Remove illegitimate break/continue/return from generated code to prevent premature loop exit."""
    lines = code.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if stripped == "break" or stripped.startswith("break ") or stripped.startswith("break#"):
            prev_lines = "\n".join(cleaned[-5:]) if cleaned else ""
            if "Error in LLM call" in prev_lines:
                cleaned.append(line)
            else:
                cleaned.append(line.replace(stripped, "pass  # sanitized: removed break"))
        elif stripped == "continue" or stripped.startswith("continue ") or stripped.startswith("continue#"):
            prev_lines = "\n".join(cleaned[-5:]) if cleaned else ""
            if "except" in prev_lines:
                cleaned.append(line)
            else:
                cleaned.append(line.replace(stripped, "pass  # sanitized: removed continue"))
        else:
            cleaned.append(line)
    return "\n".join(cleaned)

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
        raise ValueError(f"prompt must be list or str, got {type(prompt)}")
    response = _get_client().chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    content = response.choices[0].message.content
    return content if content is not None else "Output Error"


@retry(tries=5, delay=5, backoff=2, jitter=(1, 3))
def get_llm_response(messages, is_string=False, model=DEFAULT_MODEL):
    response = _get_client().chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    ans = response.choices[0].message.content
    if is_string:
        return ans
    cleaned = ans.strip("`json\n").strip("`\n").strip("```\n")
    return json.loads(cleaned)


# ── SkillModule (CONTROLLED: no LLM retrieval, uses pre-annotated skills) ──

class SkillModule:
    def __init__(self, **kwargs):
        self.skills_dir = Path(kwargs.get("skills_dir", "src/skills/webshop"))
        self.model = kwargs.get("model", DEFAULT_MODEL)

        procedure_path = kwargs.get("overall_procedure_examples_path", "")
        template_path = kwargs.get("procedure_code_template_path", "")

        self.overall_procedure_examples = ""
        if procedure_path and os.path.exists(procedure_path):
            with open(procedure_path) as f:
                self.overall_procedure_examples = f.read()

        self.procedure_code_template = ""
        if template_path and os.path.exists(template_path):
            with open(template_path) as f:
                self.procedure_code_template = f.read()

        self.metadata = self._load_metadata()

    def _load_metadata(self):
        metadata = {}
        for skill_dir in self.skills_dir.iterdir():
            if not skill_dir.is_dir():
                continue
            md_path = skill_dir / "SKILL.md"
            if not md_path.exists():
                continue
            try:
                content = md_path.read_text(encoding="utf-8")
                if content.strip().startswith("---"):
                    parts = content.split("---", 2)
                    if len(parts) >= 3:
                        header = yaml.safe_load(parts[1])
                        if isinstance(header, dict) and header.get("name") and header.get("description"):
                            metadata[header["name"]] = {
                                "description": header["description"],
                                "skill_dir": str(skill_dir),
                            }
            except Exception as e:
                print(f"[ERROR] {skill_dir.name}: {e}")
        print(f"[INFO] Loaded metadata for {len(metadata)} skills.")
        return metadata

    def generate_overall_procedure(self, task, skill_names):
        skill_contents = []
        try:
            for name in skill_names:
                if name not in self.metadata:
                    continue
                skill_dir = Path(self.metadata[name]["skill_dir"])
                if not skill_dir.is_dir():
                    continue
                combined = f"=== Skill: {name} ===\n"
                main_file = skill_dir / "SKILL.md"
                if main_file.exists():
                    combined += f"\n[File: SKILL.md]\n{main_file.read_text(encoding='utf-8')}\n"
                skill_contents.append((name, combined))
        except Exception as e:
            print(f"[ERROR] compile skill data: {e}")

        response = get_llm_response(
            generate_overall_procedure_prompt(task, self.overall_procedure_examples, skill_contents),
            is_string=True,
            model=self.model,
        )
        if "<Overall_Procedure>" in response and "</Overall_Procedure>" in response:
            return response.split("<Overall_Procedure>")[1].split("</Overall_Procedure>")[0].strip()
        if "</Analysis>" in response:
            return response.split("</Analysis>")[-1].strip()
        return response.strip()

    def generate_overall_procedure_code(self, task, overall_procedure):
        response = get_llm_response(
            generate_overall_procedure_code_prompt(task, overall_procedure, self.procedure_code_template),
            is_string=True,
            model=self.model,
        )
        pattern = r"<Overall_Procedure_Code>(.*?)</Overall_Procedure_Code>"
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            raw = matches[-1]
            if "<Overall_Procedure_Code>" in raw:
                raw = raw.split("<Overall_Procedure_Code>")[-1]
            return raw.strip().strip("```python").strip("```")
        code_match = re.search(r"```python\s*(.*?)```", response, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        return ""


# ── Colors & helpers ──

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


def run_standard_procedure(env, llm, model, messages, max_steps):
    task_done = False
    task_reward = 0
    current_steps = 0
    while not task_done and current_steps < max_steps:
        current_steps += 1
        try:
            response = llm(messages, model)
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


# ── Per-worker globals ──
_env = None
_skill_module = None
_annotations_map = None


def _init_worker(port, data_path, use_skill, model, annotations_map, init_sem=None):
    global _env, _skill_module, _client, _port, _annotations_map
    _port = port
    _client = None
    _annotations_map = annotations_map

    print(f"[Worker {os.getpid()}] port={port}, loading WebShop env...")
    _env = WebAgentTextEnv(
        observation_mode='text',
        file_path=data_path,
        human_goals=True,
        num_products=None,
    )
    print(f"[Worker {os.getpid()}] WebShop env ready.")

    if use_skill:
        _skill_module = SkillModule(
            skills_dir=os.path.join(str(PARENT), "skills"),
            overall_procedure_examples_path=os.path.join(str(WEBSHOP_ROOT), "data", "webshop_overall_procedure_examples.txt"),
            procedure_code_template_path=os.path.join(str(WEBSHOP_ROOT), "data", "webshop_procedure_code_template.py"),
            model=model,
        )


# ── Core execution (CONTROLLED: skip LLM retrieval, use annotated top-5) ──

def webshop_run_single(env, ob, instruction_text, game_idx, max_steps=30, model=None, Skill_Module=None):
    query = instruction_text
    messages = [{"role": "system", "content": webshop_system_prompt}]
    messages.append({"role": "user", "content": ob})

    task_done = False
    task_reward = 0
    steps = 0
    relevant_skill_names = []

    use_skill = False
    if Skill_Module is not None and _annotations_map is not None:
        if game_idx in _annotations_map:
            relevant_skill_names = _annotations_map[game_idx]
            if relevant_skill_names:
                use_skill = True
                print(f'{Colors.BLUE}[CONTROLLED] Using pre-annotated top-5 skills for game {game_idx}: {relevant_skill_names}{Colors.RESET}')
            else:
                print(f"[INFO] No annotated skills for game {game_idx}. Falling back to standard procedure.")
        else:
            print(f"[INFO] Game {game_idx} not in annotations. Falling back to standard procedure.")

    if use_skill:
        overall_procedure = Skill_Module.generate_overall_procedure(ob, relevant_skill_names)
        print(f'\n{Colors.BLUE}Generated Overall Procedure:\n{overall_procedure}{Colors.RESET}')

        MAX_RETRIES = 3
        retries = 0
        while retries < MAX_RETRIES:
            try:
                code = Skill_Module.generate_overall_procedure_code(ob, overall_procedure)
                print(f'\n{Colors.BLUE}Generated Code:\n{code}{Colors.RESET}')
                namespace = {}
                code = _sanitize_procedure_code(code)
                exec(code, namespace)
                func = namespace["overall_procedure_code"]
                messages, task_done, task_reward, steps = func(
                    env, llm, model, parse_action, messages, max_steps
                )
                if task_done:
                    print(f'{Colors.GREEN}Task completed! Reward: {task_reward}{Colors.RESET}')
                break
            except Exception as e:
                print(f'Error exec procedure (retry {retries+1}): {e}')
                traceback.print_exc()
                retries += 1
        else:
            print(f'{Colors.YELLOW}All {MAX_RETRIES} retries failed. Falling back to standard procedure.{Colors.RESET}')
            messages, task_done, task_reward, steps = run_standard_procedure(
                env, llm, model, messages, max_steps
            )
    else:
        messages, task_done, task_reward, steps = run_standard_procedure(
            env, llm, model, messages, max_steps
        )

    return {
        "query": query,
        "task_done": task_done,
        "reward": task_reward,
        "steps": steps,
        "messages": messages,
        "relevant_skill_names": relevant_skill_names,
    }


def eval_single_game(game_idx, session_id, args, output_path):
    global _env, _skill_module
    try:
        ob = _env.reset(session=session_id)[0]
        instruction_text = _env.instruction_text
        print(f'{Colors.RED}[Game {game_idx}] Session {session_id}: {instruction_text}{Colors.RESET}')

        result = webshop_run_single(
            env=_env,
            ob=ob,
            instruction_text=instruction_text,
            game_idx=game_idx,
            max_steps=args.max_steps,
            model=args.model,
            Skill_Module=_skill_module,
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


def load_annotations(annotations_path, top_k=5):
    """Load pre-annotated top-K skills for each game index."""
    with open(annotations_path) as f:
        data = json.load(f)
    annotations_map = {}
    for item in data:
        idx = item["index"]
        topk = [s["skill_name"] for s in item["annotation"]["skill_ranking"][:top_k]]
        annotations_map[idx] = topk
    print(f"[INFO] Loaded annotations for {len(annotations_map)} instances (top-{top_k} skills each).")
    return annotations_map


def main(args):
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    model_name = args.model.split("/")[-1]
    output_path = os.path.join(str(SCRIPT_DIR), f'results/{model_name}/{args.exp_name}_skill_{args.use_skill}')
    os.makedirs(output_path, exist_ok=True)

    test_indices_path = os.path.join(str(PARENT), 'test_indices.json')
    with open(test_indices_path) as f:
        session_ids = json.load(f)

    annotations_path = os.path.join(str(PARENT), 'webshop_skill_annotations.json')
    annotations_map = load_annotations(annotations_path, top_k=args.top_k)

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
                except:
                    continue

    tasks_to_run = {}
    for idx in range(game_start, game_end):
        if idx not in existing:
            tasks_to_run[idx] = session_ids[idx]

    print(f"Range [{game_start}, {game_end}), finished={finished_games}, remaining={len(tasks_to_run)}")
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

    def make_executor(port, n_workers):
        return concurrent.futures.ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_worker,
            initargs=(port, data_path, args.use_skill, args.model, annotations_map),
        )

    executors = []
    worker_counts = []
    for pi, port in enumerate(active_ports):
        n_workers = workers_per_port if pi < len(active_ports) - 1 else max_workers - workers_per_port * (len(active_ports) - 1)
        n_workers = max(1, n_workers)
        executors.append(make_executor(port, n_workers))
        worker_counts.append(n_workers)

    # ── True dynamic scheduling: shared queue, workers pull next task on completion ──
    pending = deque(tasks_to_run.items())
    in_flight = {}
    ex_of_future = {}
    respawned = set()

    def submit_next(ex):
        if ex in respawned or not pending:
            return
        idx, sid = pending.popleft()
        try:
            fut = ex.submit(eval_single_game, idx, sid, args, output_path)
        except concurrent.futures.process.BrokenProcessPool:
            pending.appendleft((idx, sid))
            return
        in_flight[fut] = idx
        ex_of_future[fut] = ex

    def respawn(ex):
        i = executors.index(ex)
        executors[i] = make_executor(active_ports[i], worker_counts[i])
        print(f'[Scheduler] respawned pool for port {active_ports[i]} '
              f'({worker_counts[i]} workers)')
        return executors[i]

    for ex, n in zip(executors, worker_counts):
        for _ in range(n):
            submit_next(ex)

    pbar = tqdm(total=len(tasks_to_run), desc=f"WebShop(ports={active_ports})")
    while in_flight or pending:
        if not in_flight:
            for ex, n in zip(executors, worker_counts):
                if ex not in respawned:
                    for _ in range(n):
                        submit_next(ex)
            if not in_flight:
                print(f'\n[Scheduler] no live worker pools; {len(pending)} tasks left. Aborting.')
                break
        done, _ = concurrent.futures.wait(
            list(in_flight), return_when=concurrent.futures.FIRST_COMPLETED
        )
        for future in done:
            idx = in_flight.pop(future)
            ex = ex_of_future.pop(future)
            requeued = False
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
                print(f'\nGame {idx}: worker pool crashed; requeueing task.')
                pending.appendleft((idx, tasks_to_run[idx]))
                requeued = True
                if ex not in respawned:
                    respawned.add(ex)
                    new_ex = respawn(ex)
                    for _ in range(worker_counts[executors.index(new_ex)]):
                        submit_next(new_ex)
            except Exception as exc:
                print(f'\nGame {idx} exception: {exc}')
            if not requeued:
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
        "use_skill": args.use_skill,
        "top_k": args.top_k,
        "experiment": f"text-skill-v1 topk CONTROLLED (ASIN click fix, pre-annotated top-{args.top_k} skills)",
    }
    with open(os.path.join(output_path, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--max_workers', type=int, default=12)
    parser.add_argument('--max_steps', type=int, default=30)
    parser.add_argument('--exp_name', default='qwen_webshop_textskill_v1_topk')
    parser.add_argument('--top_k', type=int, default=5,
                        help="Number of skills to slice from the annotation ranking")
    parser.add_argument('--use_skill', action='store_true')
    parser.add_argument('--ports', type=str, default="8000,8001,8002,8003")
    parser.add_argument('--game_start', type=int, default=-1)
    parser.add_argument('--game_end', type=int, default=-1)
    args = parser.parse_args()
    main(args)
