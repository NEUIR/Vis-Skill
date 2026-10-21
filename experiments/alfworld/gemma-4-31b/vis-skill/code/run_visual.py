"""
Controlled experiment — VISUAL skill, annotation-based skill selection, Gemma-4-31B.

SkillNet 3-stage pipeline with skill cards rendered as IMAGES (v5 cards).

Stage 1  retrieve_relevant_skills   -> NO LLM call: skills come from the
          pre-computed annotation alfworld_skill_ranking_per_instance.json
          (ranked by qwen3.8-max); for every instance we take the TOP-5
          skills by relevance_score.
Stage 2  generate_overall_procedure (LLM writes Procedural Guidance from the
          skill card IMAGES of the picked skills)
Stage 3  generate_overall_procedure_code (LLM implements the procedure code)
then the generated code drives the agent in the ALFWorld env, with fallback
to the standard ReAct-style loop when the skill pipeline fails.

The visual flow (system prompt, stage-2 multimodal prompt, stage-3 prompt,
VisSkillModule, agent loop, exec + fallback, result format) matches the
text-skill flow except that stage 2 receives skill card IMAGES; Stage 1 uses
the top-5 of the qwen3.8-max annotation (no LLM retrieval).

Model: Gemma-4-31B served by ONLINE vLLM servers (start_vllm.sh).
Scheduling: TRUE dynamic scheduling — all games go into one shared queue and
every worker process pulls the next game from that queue as soon as it
finishes its current one, so all GPUs stay busy until the last task.
"""
import os
import re
import json
import queue
import yaml
import base64
import argparse
import multiprocessing
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


# ── Prompt generators ──
# NOTE: retrieve_relevant_skills_prompt intentionally removed — skill
# selection is annotation-based (qwen3.8-max ranking), no LLM retrieval.

def generate_overall_procedure_prompt_visual(task, overall_procedure_examples, skill_images_b64):
    """Stage 2 prompt: multimodal with skill card images."""
    template = "You are a Senior Systems Architect."

    content = []
    content.append({
        "type": "text",
        "text": f"""You are given a complex task description, relevant skill contents (as images), and examples of well-structured procedural guidance texts.
Your task is to create a structured "Procedural Guidance" text. This text will serve as the "brain" for an AI Agent, telling it exactly how to solve the task.

# CRITICAL INSTRUCTIONS (Follow these strictly)
1.  **Structure with Phases**: You MUST divide the task into logical phases (e.g., Phase 1: Search, Phase 2: Acquire, Phase 3: Place).
2.  **Use Exact Syntax**: When describing actions, you MUST quote the exact syntax from the "Available Actions" (e.g., use "go to {{recep}}" instead of "walk to the desk").
3.  **State & Preconditions**: Explicitly mention necessary preconditions (e.g., "If the drawer is closed, use 'open drawer' before taking").
4.  **Error Handling**: Include a rule for what to do if the agent encounters an error or unexpected situation.

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

Keep your output in the format below:
<Analysis> your analysis here </Analysis>
<Overall_Procedure> your generated overall procedure guidance here </Overall_Procedure>"""
    })

    return [{"role": "system", "content": template}, {"role": "user", "content": content}]


def generate_overall_procedure_code_prompt(task, overall_procedure, procedure_code_template):
    """Stage 3 prompt: text-only (identical to SkillNet)."""
    template = "You are a Senior Software Engineer specializing in AI-driven automation and safe code generation."
    prompt = f'''
You are given a complex task description, overall procedural guidance, and a strict procedure code template.
Your task is to implement a Python function `overall_procedure_code` that follows the provided procedure code template and incorporates the overall procedure guidance to help Agent solve a complex task.

# Principles
1.  **Adherence to Template**: Strictly follow the provided procedure code template. Input arguments and return types must align with the template. (Most Important)
2.  **Inject Overall Procedural Guidance**: Help the agent accomplish the task by injecting the overall procedure guidance into the function.
4.  **Error Handling**: Implement robust error handling for LLM calls and environment interactions.
5.  **Function Naming**: Name the function `overall_procedure_code` as per the template.

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


# ── Endpoint management (per-process; port may change between tasks) ──
_port = 8000
_client_cache = {}

DEFAULT_MODEL = os.path.join(os.environ.get("MODELS_DIR", os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "..", "models"))), "Gemma-4-31B-it")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SKILL_CARDS_DIR = PROJECT_ROOT / "images" / "skill_cards"
EXAMPLES_PATH = PROJECT_ROOT / "code" / "alfworld_overall_procedure_examples.txt"
TEMPLATE_PATH = PROJECT_ROOT / "code" / "alfworld_procedure_code_template.py"
SKILL_RANKING_PATH = PROJECT_ROOT / "code" / "alfworld_skill_ranking_per_instance.json"
TOP_K = 5  # take the top-5 most relevant skills per instance from the annotation


def _set_endpoint(port):
    global _port
    _port = port


def _get_client():
    client = _client_cache.get(_port)
    if client is None:
        client = OpenAI(
            api_key="EMPTY",
            base_url=f"http://localhost:{_port}/v1",
        )
        _client_cache[_port] = client
    return client


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


# ── VisSkillModule (visual cards; annotation-based retrieval) ──

class VisSkillModule:
    def __init__(self, **kwargs):
        self.model = kwargs.get("model", DEFAULT_MODEL)
        self.skill_cards_dir = Path(kwargs.get("skill_cards_dir", str(SKILL_CARDS_DIR)))
        self.top_k = int(kwargs.get("top_k", TOP_K))

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
        for k in sorted(self.skill_images.keys())[:5]:
            print(f"  mapped: {k}")

        ranking_path = kwargs.get("skill_ranking_path", str(SKILL_RANKING_PATH))
        self.selections_by_idx, self.selections_by_name, self.task_names = \
            self._load_annotation(ranking_path)

    def _load_annotation(self, path):
        """Load the qwen3.8-max skill ranking annotation and precompute the
        top-K skill names for every ALFWorld instance (keyed both by game idx
        and by task_name)."""
        by_idx, by_name, task_names = {}, {}, {}
        if not (path and os.path.exists(path)):
            print(f"[WARN] Skill ranking file not found: {path}. No skills will be used.")
            return by_idx, by_name, task_names
        with open(path) as f:
            data = json.load(f)
        for inst in data.get("instances", []):
            ranking = sorted(
                inst.get("skill_ranking", []),
                key=lambda s: s.get("relevance_score", 0),
                reverse=True,
            )
            top = [s["skill_name"] for s in ranking[: self.top_k]]
            idx = int(inst["idx"])
            by_idx[idx] = top
            task_name = inst.get("task_name", "")
            if task_name:
                by_name[task_name] = top
                task_names[idx] = task_name
        print(f"[INFO] Loaded skill ranking annotation from {path}: "
              f"{len(by_idx)} instances, top-{self.top_k} skills each.")
        return by_idx, by_name, task_names

    def retrieve_relevant_skills(self, game_idx, task_name=None):
        """Annotation-based skill retrieval (qwen3.8-max ranking, top-K).
        NO LLM call — the selection is fully predetermined.

        Matched primarily by task_name (game_name): alfworld builds its game
        list via os.walk(), whose ordering is NOT stable across dataset
        re-syncs, so the annotation's idx order may not match the current env
        order. task_name is unique per trial, so name-based matching is exact.
        """
        if task_name and task_name in self.selections_by_name:
            expected = self.task_names.get(game_idx)
            if expected and expected != task_name:
                print(f"[INFO] game_idx={game_idx}: env game_name '{task_name}' differs "
                      f"from annotation idx-order '{expected}'; matched by game_name.")
            return list(self.selections_by_name[task_name])
        if game_idx is not None and game_idx in self.selections_by_idx:
            print(f"[WARN] game_name '{task_name}' not found in annotation; "
                  f"falling back to idx-based selection.")
            return list(self.selections_by_idx[game_idx])
        print(f"[WARN] No annotation for game_idx={game_idx} task_name={task_name}; "
              f"falling back to no skills.")
        return []

    def generate_overall_procedure(self, task, skill_names):
        """Stage 2: Generate procedure from skill card IMAGES."""
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
        """Stage 3: Generate code (text-only, identical to SkillNet)."""
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


def run_standard_procedure(env, llm, model, process_ob, messages, max_steps):
    task_done = False
    task_reward = 0
    current_steps = 0
    while not task_done and current_steps < max_steps:
        current_steps += 1
        try:
            response = llm(messages, model)
        except Exception as e:
            print(f"{Colors.RED}LLM error: {e}{Colors.RESET}")
            break
        messages.append({"role": "assistant", "content": response})
        action = parse_action(response)
        observation, task_reward, done, info = env.step([action])
        observation, task_reward, task_done = process_ob(observation[0]), info["won"][0], done[0]
        messages.append({"role": "user", "content": f"Observation: {observation}"})
        if task_done:
            print(f"{Colors.GREEN}Task completed! Reward: {task_reward}{Colors.RESET}")
    return messages, task_done, task_reward, current_steps


# ── Core execution ──

def alfworld_run_single(env, obs=[], names=[], game_idxs=None, max_steps=30, model=None,
                        Skill_Module=None):
    results = []
    for task_idx, (ob, name) in enumerate(zip(obs, names)):
        game_idx = game_idxs[task_idx] if game_idxs is not None else None
        print(f"{Colors.RED}Processing task {task_idx+1}/{len(obs)}: {name}{Colors.RESET}")
        query = ob.split("Your task is to: ")[-1].split("\n")[0].strip()
        messages = [{"role": "system", "content": alfworld_system_prompt}]
        messages.append({"role": "user", "content": ob})

        task_done = False
        task_reward = 0
        steps = 0
        relevant_skill_names = []

        use_skill = False
        if Skill_Module is not None:
            relevant_skill_names = Skill_Module.retrieve_relevant_skills(game_idx, name)
            if relevant_skill_names:
                use_skill = True
                print(f"{Colors.BLUE}Annotated top-{Skill_Module.top_k} skills: {relevant_skill_names}{Colors.RESET}")
            else:
                print(f"[INFO] No annotated skills for: {name}. Falling back.")

        if use_skill:
            overall_procedure = Skill_Module.generate_overall_procedure(ob, relevant_skill_names)
            print(f"\n{Colors.BLUE}Procedure:\n{overall_procedure[:200]}...{Colors.RESET}")

            MAX_RETRIES = 3
            retries = 0
            skill_success = False
            while retries < MAX_RETRIES:
                try:
                    code = Skill_Module.generate_overall_procedure_code(ob, overall_procedure)
                    if not code.strip():
                        raise ValueError("Empty code generated")
                    namespace = {}
                    exec(code, namespace)
                    func = namespace["overall_procedure_code"]
                    messages, task_done, task_reward, steps = func(
                        env, llm, model, process_ob, parse_action, messages, max_steps
                    )
                    if task_done:
                        print(f"{Colors.GREEN}Task completed! Reward: {task_reward}{Colors.RESET}")
                    skill_success = True
                    break
                except Exception as e:
                    print(f"Error exec procedure (retry {retries+1}): {e}")
                    retries += 1

            if not skill_success:
                print(f"{Colors.YELLOW}Skill pipeline failed, falling back to standard procedure.{Colors.RESET}")
                messages, task_done, task_reward, steps = run_standard_procedure(
                    env, llm, model, process_ob, messages, max_steps
                )
        else:
            messages, task_done, task_reward, steps = run_standard_procedure(
                env, llm, model, process_ob, messages, max_steps
            )

        # Strip image content from messages before saving (keep text only)
        clean_messages = []
        for msg in messages:
            if isinstance(msg.get("content"), list):
                text_parts = [p["text"] for p in msg["content"] if p.get("type") == "text"]
                clean_messages.append({"role": msg["role"], "content": "\n".join(text_parts)})
            else:
                clean_messages.append(msg)

        results.append({
            "query": query,
            "name": name,
            "task_done": task_done,
            "reward": task_reward,
            "steps": steps,
            "messages": clean_messages,
            "relevant_skill_names": relevant_skill_names,
            "skill_selection_source": (
                "alfworld_skill_ranking_per_instance.json (qwen3.8-max annotation, "
                f"top-{Skill_Module.top_k} by relevance_score)"
                if Skill_Module is not None else "none"
            ),
        })
    return results


def eval_single_game(game_idx, args, config, split, output_path):
    env = None
    try:
        env = get_environment(config["env"]["type"])(config, train_eval=split)
        env = env.init_env(batch_size=1)
        for _ in range(game_idx + 1):
            obs_list, info = env.reset()

        ob_str = "\n".join(obs_list[0].split("\n\n")[1:])
        game_name = "/".join(info["extra.gamefile"][0].split("/")[-3:-1])

        Skill_Module = None
        if args.use_skill:
            Skill_Module = VisSkillModule(
                skill_cards_dir=str(SKILL_CARDS_DIR),
                overall_procedure_examples_path=str(EXAMPLES_PATH),
                procedure_code_template_path=str(TEMPLATE_PATH),
                skill_ranking_path=str(SKILL_RANKING_PATH),
                top_k=args.top_k,
                model=args.model,
            )

        batch_results = alfworld_run_single(
            env=env,
            obs=[ob_str],
            names=[game_name],
            game_idxs=[game_idx],
            max_steps=args.max_steps,
            model=args.model,
            Skill_Module=Skill_Module,
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


# ── Dynamic-scheduling worker ──
#
# Every worker process loops: pull the next game index from ONE shared queue,
# round-robin assign it to a vLLM port (shared atomic counter), run it, report
# the result. No static partitioning: whichever worker frees up first takes
# the next task, so all GPUs stay busy until the queue is empty.

def _worker_loop(task_queue, result_queue, ports, port_counter, args, config, split, output_path):
    while True:
        idx = task_queue.get()
        if idx is None:  # sentinel -> shut down
            break
        with port_counter.get_lock():
            order = port_counter.value
            port_counter.value += 1
        port = ports[order % len(ports)]
        _set_endpoint(port)
        try:
            result = eval_single_game(idx, args, config, split, output_path)
        except Exception as exc:
            print(f"Game {idx} exception: {exc}")
            import traceback; traceback.print_exc()
            result = None
        result_queue.put((idx, result))


# ── Summary ──

def generate_summary(output_path, summary_path, model):
    results = []
    for f in sorted(Path(output_path).glob("idx_*.json"), key=lambda p: int(p.stem.split("_")[1])):
        with open(f) as fh:
            results.append(json.load(fh))

    if not results:
        print("[WARN] No results to summarize.")
        return

    def classify_task(name):
        n = name.lower()
        if "pick_two" in n: return "pick_two"
        if "pick_and_place" in n: return "pick_place"
        if "pick_clean" in n: return "pick_clean"
        if "pick_heat" in n: return "pick_heat"
        if "pick_cool" in n: return "pick_cool"
        if "look_at" in n or "examine" in n: return "examine"
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

    model_name = model.split("/")[-1]
    summary = {
        "experiment": "alfworld-gemma-controlled-qwen3.8max-31B-vis-skill",
        "description": (f"Visual skill cards, skills fixed to top-{TOP_K} of the "
                        f"qwen3.8-max annotation (alfworld_skill_ranking_per_instance.json)"),
        "model": model_name,
        "overall": {
            "total": total,
            "success": success,
            "success_rate": round(success / total * 100, 2) if total else 0,
            "avg_steps": round(total_steps / total, 2) if total else 0,
        },
        "by_task_type": {},
    }

    for t in ["pick_place", "pick_clean", "pick_heat", "pick_cool", "pick_two", "examine"]:
        if t in by_type:
            d = by_type[t]
            summary["by_task_type"][t] = {
                "total": d["total"],
                "success": d["success"],
                "success_rate": round(d["success"] / d["total"] * 100, 2) if d["total"] else 0,
                "avg_steps": round(d["steps"] / d["total"], 2) if d["total"] else 0,
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


def main(args):
    config_path = Path(__file__).resolve().parent / "base_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    split = "eval_out_of_distribution"

    model_name = args.model.split("/")[-1]
    output_path = f"{PROJECT_ROOT}/results/alfworld/{model_name}/vis_baseline"
    os.makedirs(output_path, exist_ok=True)

    temp_env = get_environment(config["env"]["type"])(config, train_eval=split)
    temp_env = temp_env.init_env(batch_size=1)
    total_games = len(temp_env.gamefiles)
    del temp_env

    game_start = args.game_start if args.game_start >= 0 else 0
    game_end = args.game_end if args.game_end > 0 else total_games
    game_end = min(game_end, total_games)
    print(f"Total games: {total_games}, running [{game_start}, {game_end})")

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
                except Exception:
                    continue

    tasks_to_run = [i for i in range(game_start, game_end) if i not in existing]
    print(f"Already finished: {finished_games}, Remaining: {len(tasks_to_run)}")

    if tasks_to_run:
        ports = [int(p.strip()) for p in args.ports.split(",")]
        num_workers = max(1, args.max_workers)
        print(f"Dynamic scheduling: {num_workers} workers pulling from ONE shared queue, "
              f"round-robin over vLLM ports {ports} ...")

        ctx = multiprocessing.get_context("fork")
        task_queue = ctx.Queue()
        result_queue = ctx.Queue()
        port_counter = ctx.Value("i", 0)

        for idx in tasks_to_run:
            task_queue.put(idx)
        for _ in range(num_workers):
            task_queue.put(None)  # shutdown sentinels

        procs = []
        for w in range(num_workers):
            p = ctx.Process(
                target=_worker_loop,
                args=(task_queue, result_queue, ports, port_counter,
                      args, config, split, output_path),
                daemon=True,
            )
            p.start()
            procs.append(p)

        pbar = tqdm(total=len(tasks_to_run), desc=f"Eval(ports={ports})")
        received = 0
        while received < len(tasks_to_run):
            try:
                idx, result = result_queue.get(timeout=7200)
            except queue.Empty:
                print(f"\n[WARN] No result for 7200s; {len(tasks_to_run)-received} tasks outstanding. Stopping.")
                break
            received += 1
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
            pbar.update(1)
        pbar.close()

        for p in procs:
            p.join(timeout=60)
        for p in procs:
            if p.is_alive():
                p.terminate()

    print(f"\nDone. Finished: {finished_games}")
    if finished_games > 0:
        print(f"Avg Reward: {all_rewards/finished_games:.4f}")
        print(f"Avg Steps: {all_steps/finished_games:.1f}")

    all_done = all(
        (Path(output_path) / f"idx_{i}.json").exists()
        for i in range(game_start, game_end)
    )
    if all_done:
        summary_path = f"{output_path}/summary_vis_baseline.json"
        generate_summary(output_path, summary_path, args.model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max_workers", type=int, default=48,
                        help="number of dynamic worker processes (high concurrency)")
    parser.add_argument("--max_steps", type=int, default=30)
    parser.add_argument("--use_skill", action="store_true")
    parser.add_argument("--top_k", type=int, default=TOP_K,
                        help="number of top-ranked annotated skills per instance")
    parser.add_argument("--port", type=int, default=8000,
                        help="(legacy, unused) kept for reference compatibility")
    parser.add_argument("--ports", type=str, default="8000,8001,8002,8003")
    parser.add_argument("--game_start", type=int, default=-1)
    parser.add_argument("--game_end", type=int, default=-1)
    args = parser.parse_args()
    main(args)
