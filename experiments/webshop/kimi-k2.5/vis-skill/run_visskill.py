"""
Vis-skill WebShop evaluation via DashScope API (multimodal).
CONTROLLED variant:
  1. FIXED skill selections from QWEN3.8-MAX ANNOTATIONS (skill_selections.json)
  2. VLM generates phased procedural guidance from rendered skill card IMAGES
  3. LLM generates Python code (text-only) -> exec() runs the agent loop
     (fallback to standard loop on failure, 3 retries)

The ONLY difference from text-skill is Stage 2 input modality: skill content
arrives as rendered PNG images instead of raw SKILL.md text.

Execution model: kimi-k2.5 via the DashScope API.
Dynamic scheduling — all games go into one shared queue.
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
import base64
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

SKILLS_DIR = SCRIPT_DIR / "skills"
SKILL_CARDS_DIR = SCRIPT_DIR / "images" / "skill_cards_v7"
EXAMPLES_PATH = SHARED_DATA_DIR / "webshop_overall_procedure_examples.txt"
TEMPLATE_PATH = SHARED_DATA_DIR / "webshop_procedure_code_template.py"
SELECTIONS_PATH = SCRIPT_DIR / "skill_selections.json"

_client = None
_TEMPERATURE = None


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


@retry(tries=5, delay=5, backoff=2, jitter=(1, 3))
def get_llm_response(messages, is_string=False, model=DEFAULT_MODEL):
    response = _get_client().chat.completions.create(
        model=model,
        messages=messages,
        **_sampling_kwargs(),
    )
    ans = response.choices[0].message.content
    if is_string:
        return ans
    cleaned = ans.strip("`json\n").strip("`\n").strip("```\n")
    return json.loads(cleaned)


def generate_overall_procedure_prompt_visual(task, overall_procedure_examples, skill_images_b64):
    template = "You are a Senior Systems Architect."

    content = []
    content.append({
        "type": "text",
        "text": f"""You are given a complex task description, relevant skill contents (as images), and examples of well-structured procedural guidance texts.
Your task is to create a structured "Procedural Guidance" text. This text will serve as the "brain" for an AI Agent, telling it exactly how to solve the task.

# CRITICAL INSTRUCTIONS (Follow these strictly)
1.  **Structure with Phases**: You MUST divide the task into logical phases (e.g., Phase 1: Search, Phase 2: Selection, Phase 3: Configuration, Phase 4: Purchase).
2.  **Use Exact Syntax**: When describing actions, you MUST quote the exact syntax from the "Available Actions" (e.g., use "search[keywords]" or "click[value]").
3.  **State & Preconditions**: Explicitly mention necessary preconditions (e.g., "Verify the price is below the target before purchasing").
4.  **Error Handling**: Include a rule for what to do if the agent encounters an error or unexpected situation.
5.  **Product Click Format (MANDATORY)**: When instructing the agent to open a product from the search results page, ALWAYS use the product's ASIN code (e.g., `click[B07WSW2FTQ]`). The ASIN is the short alphanumeric code (starts with B0) shown before each product title in the search results. Product titles, names, and descriptions are NOT clickable — `click[<item title>]` or `click[<product name>]` is an INVALID action and the page will not change.

# Input Data
1.  **Task Description:**
{task}
2.  **Relevant Skill Contents (see images below):**"""
    })

    for name, b64 in skill_images_b64:
        content.append({"type": "text", "text": f"\n[Skill: {name}]"})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"}
        })

    content.append({
        "type": "text",
        "text": f"""

# Examples
Here are some examples of well-structured Procedural Guidance texts you should emulate (focus on style, structure, and level of detail instead of content):
{overall_procedure_examples}

# IMPORTANT CORRECTION TO THE EXAMPLES ABOVE
The examples above come from a related environment and some of them describe clicking a product by its full title or name (for example `click[{{item_title}}]`, `click[<item_name>]`, or `click[<Full Name of Item>]`).
That format is INVALID in THIS WebShop environment: clicking a product title produces an invalid action and the page does not change, wasting steps until timeout.
In THIS environment, every product on the search results page is preceded by its ASIN code — a short alphanumeric identifier (starts with B0) placed before the product title (e.g., B08BTJ1N6V). The ASIN code is the ONLY valid click target for opening a product page.
Therefore, in the Procedural Guidance you generate:
- ALWAYS instruct the agent to use `click[<ASIN>]` (e.g., `click[B08BTJ1N6V]`) to open a product from search results.
- NEVER instruct the agent to click a product title, name, or description.
- Include an error-handling rule: if an action had no effect (the observation did not change), the clicked target was invalid; the agent must re-read the observation and click the ASIN code instead.

Keep your output in the format below:
<Analysis> your analysis here </Analysis>
<Overall_Procedure> your generated overall procedure guidance here </Overall_Procedure>"""
    })

    return [{"role": "system", "content": template}, {"role": "user", "content": content}]


def generate_overall_procedure_code_prompt(task, overall_procedure, procedure_code_template):
    template = "You are a Senior Software Engineer specializing in AI-driven automation and safe code generation."
    prompt = f'''
You are given a complex task description, overall procedural guidance, and a strict procedure code template.
Your task is to implement a Python function `overall_procedure_code` that follows the provided procedure code template and incorporates the overall procedure guidance to help Agent solve a complex task.

# Principles
1.  **Adherence to Template**: Strictly follow the provided procedure code template. Input arguments and return types must align with the template. (Most Important)
2.  **Inject Overall Procedural Guidance**: Help the agent accomplish the task by injecting the overall procedure guidance into the function.
4.  **Error Handling**: Implement robust error handling for LLM calls and environment interactions.
5.  **Function Naming**: Name the function `overall_procedure_code` as per the template.
6.  **Invalid Action Detection & Correction**: In the template's "RUNTIME ANALYSIS & INTERVENTION" section, implement stuck detection: keep track of the previous observation (a variable initialized before the loop), and when the current observation is exactly identical to the previous one, the last action was INVALID and had no effect — in that case append a user-message hint telling the agent: the previous action was invalid; on search result pages each product is preceded by its ASIN code (a short code starting with B0, e.g., B08BTJ1N6V), and the agent must click that ASIN code (e.g., `click[B08BTJ1N6V]`), NOT the product title or description. Do NOT use `continue`, `break`, or `return` in that section.

# Input Data
1.  **Task Description:**
{task}
2.  **Overall Procedural Guidance:**
{overall_procedure}
3.  **Procedure Code Template:**
```python
{procedure_code_template}
```

Keep your output in the format below:
<Analysis> your analysis here </Analysis>
<Overall_Procedure_Code> your generated overall procedure workflow code here </Overall_Procedure_Code>
'''
    return [{"role": "system", "content": template}, {"role": "user", "content": prompt}]


class VisSkillModule:
    def __init__(self, **kwargs):
        self.model = kwargs.get("model", DEFAULT_MODEL)
        self.skill_cards_dir = Path(kwargs.get("skill_cards_dir", str(SKILL_CARDS_DIR)))

        examples_path = kwargs.get("overall_procedure_examples_path", "")
        template_path = kwargs.get("procedure_code_template_path", "")

        self.overall_procedure_examples = ""
        if examples_path and os.path.exists(examples_path):
            with open(examples_path) as f:
                self.overall_procedure_examples = f.read()

        self.procedure_code_template = ""
        if template_path and os.path.exists(template_path):
            with open(template_path) as f:
                self.procedure_code_template = f.read()

        self.skill_images = {}
        for img_path in sorted(self.skill_cards_dir.glob("*.png")):
            skill_name = img_path.stem
            with open(img_path, "rb") as f:
                self.skill_images[skill_name] = base64.b64encode(f.read()).decode("utf-8")
        print(f"[INFO] Loaded {len(self.skill_images)} skill card images.")

    def generate_overall_procedure(self, task, skill_names):
        skill_images_b64 = []
        for name in skill_names:
            if name in self.skill_images:
                skill_images_b64.append((name, self.skill_images[name]))
            else:
                print(f"[WARN] No image for skill: {name}")

        if not skill_images_b64:
            return ""

        response = get_llm_response(
            generate_overall_procedure_prompt_visual(
                task, self.overall_procedure_examples, skill_images_b64
            ),
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
        return match.group(1).strip().strip('"\'\'*`')
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


_skill_selections = None


def _load_selections():
    global _skill_selections
    if _skill_selections is None:
        with open(SELECTIONS_PATH) as f:
            _skill_selections = json.load(f)
    return _skill_selections


_env = None
_skill_module = None


def _init_worker(counter, data_path, use_skill, model, init_sem=None):
    global _env, _skill_module, _client
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

        if use_skill:
            _skill_module = VisSkillModule(
                skill_cards_dir=str(SKILL_CARDS_DIR),
                overall_procedure_examples_path=str(EXAMPLES_PATH),
                procedure_code_template_path=str(TEMPLATE_PATH),
                model=model,
            )


def webshop_run_single(env, ob, instruction_text, max_steps=30, model=None,
                       Skill_Module=None, fixed_skills=None):
    query = instruction_text
    messages = [{"role": "system", "content": webshop_system_prompt}]
    messages.append({"role": "user", "content": ob})

    task_done = False
    task_reward = 0
    steps = 0
    relevant_skill_names = fixed_skills if fixed_skills else []

    use_skill = bool(relevant_skill_names) and Skill_Module is not None
    if use_skill:
        print(f'{Colors.BLUE}Using fixed skills: {relevant_skill_names}{Colors.RESET}')
    else:
        print(f"[INFO] No skills for this task. Falling back to standard procedure.")

    if use_skill:
        overall_procedure = Skill_Module.generate_overall_procedure(ob, relevant_skill_names)
        print(f'\n{Colors.BLUE}Generated Overall Procedure:\n{overall_procedure[:300]}...{Colors.RESET}')

        MAX_RETRIES = 3
        retries = 0
        skill_success = False
        while retries < MAX_RETRIES:
            try:
                code = Skill_Module.generate_overall_procedure_code(ob, overall_procedure)
                if not code.strip():
                    raise ValueError("Empty code generated")
                print(f'\n{Colors.BLUE}Generated Code:\n{code[:200]}...{Colors.RESET}')
                namespace = {}
                code = _sanitize_procedure_code(code)
                exec(code, namespace)
                func = namespace["overall_procedure_code"]
                messages, task_done, task_reward, steps = func(
                    env, llm, model, parse_action, messages, max_steps
                )
                if task_done:
                    print(f'{Colors.GREEN}Task completed! Reward: {task_reward}{Colors.RESET}')
                skill_success = True
                break
            except Exception as e:
                print(f'Error exec procedure (retry {retries+1}): {e}')
                traceback.print_exc()
                retries += 1

        if not skill_success:
            print(f'{Colors.YELLOW}Skill pipeline failed, falling back to standard procedure.{Colors.RESET}')
            messages, task_done, task_reward, steps = run_standard_procedure(
                env, llm, model, messages, max_steps
            )
    else:
        messages, task_done, task_reward, steps = run_standard_procedure(
            env, llm, model, messages, max_steps
        )

    clean_messages = []
    for msg in messages:
        if isinstance(msg.get("content"), list):
            text_parts = [p["text"] for p in msg["content"] if p.get("type") == "text"]
            clean_messages.append({"role": msg["role"], "content": "\n".join(text_parts)})
        else:
            clean_messages.append(msg)

    return {
        "query": query,
        "task_done": task_done,
        "reward": task_reward,
        "steps": steps,
        "messages": clean_messages,
        "relevant_skill_names": relevant_skill_names,
    }


def eval_single_game(game_idx, session_id, args, output_path):
    global _env, _skill_module
    try:
        selections = _load_selections()
        fixed_skills = selections.get(str(game_idx), {}).get("relevant_skill_names", [])

        ob = _env.reset(session=session_id)[0]
        instruction_text = _env.instruction_text
        print(f'{Colors.RED}[Game {game_idx}] Session {session_id}: {instruction_text}{Colors.RESET}')

        result = webshop_run_single(
            env=_env,
            ob=ob,
            instruction_text=instruction_text,
            max_steps=args.max_steps,
            model=args.model,
            Skill_Module=_skill_module,
            fixed_skills=fixed_skills,
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

    if args.use_skill and not SELECTIONS_PATH.exists():
        print(f"[ERROR] {SELECTIONS_PATH} not found.")
        sys.exit(1)

    model_name = args.model.split("/")[-1]
    output_path = os.path.join(str(SCRIPT_DIR), f'results/{model_name}/{args.exp_name}_skill_{args.use_skill}')
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

    def make_executor(counter):
        # NOTE: semaphore created PER POOL — if a worker dies while holding a
        # pool-shared semaphore, the poisoned semaphore would deadlock every
        # respawned pool as well. A fresh semaphore per
        # pool keeps respawns working.
        init_sem = multiprocessing.Semaphore(1)
        return concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_init_worker,
            initargs=(counter, data_path, args.use_skill, args.model, init_sem),
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

        pbar = tqdm(total=len(remaining), desc=f"WebShop-VisSkill({model_name}, DashScope API)")
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
        "use_skill": args.use_skill,
        "experiment": "vis-skill CONTROLLED (skill cards as images, top-5 from Qwen3.8-Max annotations)",
    }
    with open(os.path.join(output_path, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--max_workers', type=int, default=20,
                        help="number of dynamic worker processes")
    parser.add_argument('--max_steps', type=int, default=30)
    parser.add_argument('--exp_name', default='webshop_visskill')
    parser.add_argument('--use_skill', action='store_true')
    parser.add_argument('--temperature', type=float, default=None,
                        help="sampling temperature (default: not passed, API default)")
    parser.add_argument('--game_start', type=int, default=-1)
    parser.add_argument('--game_end', type=int, default=-1)
    args = parser.parse_args()
    main(args)
