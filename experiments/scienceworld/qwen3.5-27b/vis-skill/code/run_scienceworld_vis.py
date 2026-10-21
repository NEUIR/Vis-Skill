"""
ScienceWorld VIS-SKILL CONTROLLED experiment (backbone: Qwen3.5-27B).

Controlled counterpart of ../text-skill: identical pipeline except that the
selected skills are presented to the model as RENDERED SKILL-CARD IMAGES
(renderer v6: code/render_skill_cards.py, structure-preserving, 32px-grid
aligned for Qwen3.5 vision) instead of SKILL.md text during stage 2
(procedure generation). The skill cards were rendered ahead of time and are
copied into images/skill_cards_v6/ (named by SKILL.md frontmatter name).

Guarantees:
  - The skills used for every episode are the CONTROLLED selections: the
    qwen3.8-max pre-annotation top-5 per episode — exactly the same fixed
    selections the text-skill run uses. They are loaded from
    skill_selections.json (produced by code/prepare_skill_selections.py from
    science_skill_annotations.json, with completeness + consistency checks
    and a cross-check against the text-skill results). No re-retrieval
    happens here.
  - Everything else (system prompt, stage-2/3 prompts structure, code
    template, exec retries, fallback, env protocol, max_steps) is unchanged
    from the text-skill experiment. Only the skill rendering form differs.

Inherited v1 robustness (same as text-skill):
  - No ProcessPoolExecutor.future.result() without timeout. Every episode
    runs in its own grandchild process joined with a HARD wall-clock limit
    (--episode_timeout); on timeout the whole process group (python +
    ScienceWorld JVM) is SIGKILLed, the episode retried (--max_attempts)
    and, if still failing, recorded as an error result. Workers always
    return to the shared queue.
  - exec()'d LLM code infinite loops / env.step() on a dead JVM / vLLM
    drops after @retry exhaustion: all covered by the same mechanisms.

Scheduling is fully DYNAMIC: ONE shared queue for all episodes; each of the
N persistent workers pulls the NEXT episode the moment it finishes its
current one, and every LLM request round-robins across ALL online vLLM
servers (ports --base_port .. --base_port+--num_ports-1), so every GPU
stays busy until the very last task.
"""
import os
import re
import sys
import json
import time
import base64
import queue as _queue_mod
import signal
import argparse
import threading
import traceback
import multiprocessing
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from retry import retry
import httpx
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


# ── Prompt generators ──

def generate_overall_procedure_prompt_visual(task, overall_procedure_examples, skill_images_b64):
    """Stage 2 prompt: multimodal with skill card images
    (structure identical to ../vis-skill/code/run_scienceworld_vis.py)."""
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
    """Stage 3 prompt: text-only (identical to SkillNet / text-skill)."""
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


# ── Globals (set inside each episode process) ──

DEFAULT_MODEL = os.path.join(os.environ.get("MODELS_DIR", os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "..", "models"))), "Qwen3.5-27B")

CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[1]   # vis-skill/ (controlled)
INDICES_PATH = CODE_DIR / "test_indices.json"
OVERALL_PROCEDURE_EXAMPLES_PATH = CODE_DIR / "scienceworld_overall_procedure_examples.txt"
PROCEDURE_CODE_TEMPLATE_PATH = CODE_DIR / "scienceworld_procedure_code_template.py"
SKILL_CARDS_DIR = PROJECT_ROOT / "images" / "skill_cards_v6"
SELECTIONS_PATH = PROJECT_ROOT / "skill_selections.json"

_BASE_PORT = 8000
_NUM_PORTS = 2
_CLIENT_TIMEOUT = 300.0

_client_cache = {}
# Randomized per episode process (in _episode_entry) so that phase-aligned
# requests from many concurrent episodes spread evenly over all vLLM ports
# instead of stampeding the same port.
_rr_counter = 0
_indices_cache = None


def _indices():
    global _indices_cache
    if _indices_cache is None:
        with open(INDICES_PATH) as f:
            _indices_cache = json.load(f)
    return _indices_cache


def _get_client():
    """Round-robin across ALL online vLLM servers, per request (dynamic)."""
    global _rr_counter
    port = _BASE_PORT + (_rr_counter % _NUM_PORTS)
    _rr_counter += 1
    client = _client_cache.get(port)
    if client is None:
        client = OpenAI(
            api_key="EMPTY",
            base_url=f"http://localhost:{port}/v1",
            timeout=httpx.Timeout(_CLIENT_TIMEOUT, connect=10.0),
        )
        _client_cache[port] = client
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
    if ans is None:
        ans = "Output Error"
    if is_string:
        return ans
    cleaned = ans.strip("`json\n").strip("`\n").strip("```\n")
    return json.loads(cleaned)


# ── VisSkillModule (fixed selections + skill-card images) ──

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
        print(f"[INFO] Loaded {len(self.skill_images)} skill card images from {self.skill_cards_dir}.")

    def generate_overall_procedure(self, task, skill_names):
        """Stage 2: generate procedure from skill card IMAGES (SKILL.md content, rendered)."""
        skill_images_b64 = []
        for name in skill_names:
            if name in self.skill_images:
                skill_images_b64.append((name, self.skill_images[name]))
            else:
                print(f"[WARN] No skill card image for: {name}")

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
        """Stage 3: generate code (text-only, identical to text-skill)."""
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


# ── Fixed skill selections (controlled: qwen3.8-max annotation top-5) ──

_selections_cache = None


def _selections():
    global _selections_cache
    if _selections_cache is None:
        if not SELECTIONS_PATH.exists():
            raise FileNotFoundError(
                f"{SELECTIONS_PATH} not found. Run code/prepare_skill_selections.py "
                f"(annotation-driven) before the vis-skill experiment."
            )
        with open(SELECTIONS_PATH) as f:
            _selections_cache = json.load(f)
    return _selections_cache


def _fixed_skills_for(idx, task_name, var_idx):
    entry = _selections().get(str(idx))
    if entry is None:
        # In the controlled experiment selections come from the complete
        # qwen3.8-max annotation (all 211 episodes), so a missing entry is a
        # hard error, not a silent no-skill fallback.
        raise RuntimeError(
            f"skill_selections.json has no entry for episode {idx} "
            f"({task_name} var {var_idx}) — run code/prepare_skill_selections.py"
        )
    # Consistency guard: the selection must belong to exactly this episode.
    if entry.get("task_name") != task_name or int(entry.get("variation_idx", -1)) != int(var_idx):
        raise RuntimeError(
            f"skill_selections.json entry {idx} mismatch: "
            f"expected {task_name}/var{var_idx}, "
            f"got {entry.get('task_name')}/var{entry.get('variation_idx')}"
        )
    return entry.get("relevant_skill_names", [])


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


# ── Core execution (identical pipeline to ../vis-skill) ──

def scienceworld_run_single(env, task_name, var_idx, args, Vis_Skill_Module=None, fixed_skills=None):
    obs, info = env.reset()
    query = env.get_task_description()

    print(f"{Colors.RED}Processing: {task_name} (Var: {var_idx}){Colors.RESET}")
    print(f"Query: {query}")

    messages = [{"role": "system", "content": scienceworld_system_prompt}]
    messages.append({"role": "user", "content": query})

    task_done = False
    task_reward = 0
    steps = 0
    relevant_skill_names = fixed_skills if fixed_skills else []

    use_skill = bool(relevant_skill_names) and Vis_Skill_Module is not None
    if use_skill:
        print(f"{Colors.BLUE}Using fixed skills (qwen3.8-max annotation, controlled): "
              f"{relevant_skill_names}{Colors.RESET}")

    if use_skill:
        overall_procedure = Vis_Skill_Module.generate_overall_procedure(query, relevant_skill_names)
        print(f"\n{Colors.BLUE}Procedure:\n{overall_procedure[:200]}...{Colors.RESET}")

        MAX_RETRIES = 3
        retries = 0
        skill_success = False
        while retries < MAX_RETRIES:
            try:
                code = Vis_Skill_Module.generate_overall_procedure_code(query, overall_procedure)
                if not code.strip():
                    raise ValueError("Empty code generated")
                namespace = {}
                exec(code, namespace)
                func = namespace.get("overall_procedure_code")
                if func is None:
                    raise ValueError("Function 'overall_procedure_code' not found in generated code.")
                messages, task_done, task_reward, steps = func(
                    env, llm, args.model, parse_action, messages, args.max_steps
                )
                if task_done:
                    print(f"{Colors.GREEN}Task completed! Score: {task_reward}{Colors.RESET}")
                skill_success = True
                break
            except Exception as e:
                print(f"Error exec procedure (retry {retries + 1}): {e}")
                traceback.print_exc()
                retries += 1

        if not skill_success:
            print(f"{Colors.YELLOW}Skill pipeline failed, falling back to standard procedure.{Colors.RESET}")
            messages, task_done, task_reward, steps = run_standard_procedure(
                env, llm, args.model, messages, args.max_steps
            )
    else:
        messages, task_done, task_reward, steps = run_standard_procedure(
            env, llm, args.model, messages, args.max_steps
        )

    # strip multimodal content for JSON storage
    clean_messages = []
    for msg in messages:
        if isinstance(msg.get("content"), list):
            text_parts = [p["text"] for p in msg["content"] if p.get("type") == "text"]
            clean_messages.append({"role": msg["role"], "content": "\n".join(text_parts)})
        else:
            clean_messages.append(msg)

    return {
        "query": query,
        "name": task_name,
        "variation_idx": var_idx,
        "task_done": task_done,
        "reward": task_reward,
        "steps": steps,
        "messages": clean_messages,
        "relevant_skill_names": relevant_skill_names,
    }


# ── Episode process (one fresh process + one fresh JVM per episode) ──

def _kill_env_java(env):
    """Force-kill the ScienceWorld JVM without ever waiting on it."""
    try:
        gw = getattr(env, "_gateway", None)
        proc = getattr(gw, "java_process", None) if gw is not None else None
        if proc is not None and proc.poll() is None:
            proc.kill()
    except Exception:
        pass


def _episode_entry(idx, attempt, worker_id, cfg):
    """Runs one episode end-to-end and writes idx_<idx>.json on success.

    Runs in its own session (os.setsid) so the supervising worker can kill
    the ENTIRE process tree (python + JVM) with one killpg() on timeout.
    """
    try:
        os.setsid()
    except OSError:
        pass

    global _BASE_PORT, _NUM_PORTS, _CLIENT_TIMEOUT, _rr_counter
    _BASE_PORT = cfg["base_port"]
    _NUM_PORTS = cfg["num_ports"]
    _CLIENT_TIMEOUT = cfg["client_timeout"]
    # Random round-robin start offset: concurrent episodes fan out over all
    # vLLM servers instead of hitting the same port in phase.
    _rr_counter = int.from_bytes(os.urandom(2), "little")

    task_name, var_idx = "?", -1
    env = None
    try:
        task_name, var_idx = _indices()[idx]
        print(f"[ep {idx} att {attempt} wid {worker_id}] start {task_name} var {var_idx}", flush=True)

        fixed_skills = _fixed_skills_for(idx, task_name, var_idx)

        vis_skill_module = None
        if cfg["use_skill"]:
            vis_skill_module = VisSkillModule(
                skill_cards_dir=cfg["skill_cards_dir"],
                overall_procedure_examples_path=cfg["overall_procedure_examples_path"],
                procedure_code_template_path=cfg["procedure_code_template_path"],
                model=cfg["model"],
            )

        env = ScienceWorldEnv(taskName=None, serverPath=None, envStepLimit=cfg["max_steps"])
        env.load(task_name, var_idx, simplificationStr="easy")

        args_ns = argparse.Namespace(model=cfg["model"], max_steps=cfg["max_steps"])
        result = scienceworld_run_single(
            env, task_name, var_idx, args_ns, vis_skill_module, fixed_skills
        )

        save_file = os.path.join(cfg["output_path"], f"idx_{idx}.json")
        tmp_file = f"{save_file}.tmp.{os.getpid()}"
        with open(tmp_file, "w") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        os.replace(tmp_file, save_file)
        print(f"[ep {idx} att {attempt}] DONE score={result.get('reward')} "
              f"steps={result.get('steps')} done={result.get('task_done')}", flush=True)
    except Exception as e:
        print(f"[ep {idx} att {attempt}] ERROR ({task_name} var {var_idx}): {e}", flush=True)
        traceback.print_exc()
        os._exit(1)
    finally:
        if env is not None:
            _kill_env_java(env)
    os._exit(0)


# ── Process-tree kill (timeout enforcement) ──

def _kill_tree(pid):
    """SIGKILL a process and all of its descendants (JVM included)."""
    victims = {pid}
    try:
        import psutil
        p = psutil.Process(pid)
        victims.update(c.pid for c in p.children(recursive=True))
    except Exception:
        pass
    try:
        os.killpg(pid, signal.SIGKILL)   # works because episode called os.setsid()
    except Exception:
        pass
    for v in victims:
        try:
            os.kill(v, signal.SIGKILL)
        except Exception:
            pass


def _result_path(cfg, idx):
    return Path(cfg["output_path"]) / f"idx_{idx}.json"


# ── Worker: persistent, pulls next episode from the shared queue ──

def _worker_loop(worker_id, task_q, status_q, cfg):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    ctx = multiprocessing.get_context("spawn")

    while True:
        item = task_q.get()
        if item is None:
            break
        idx, attempt = item
        status_q.put(("took", worker_id, idx, attempt))

        try:
            child = ctx.Process(
                target=_episode_entry, args=(idx, attempt, worker_id, cfg), daemon=False
            )
            child.start()
        except Exception as e:
            status_q.put(("failed", worker_id, idx, attempt, f"spawn-error: {e}"))
            continue

        status_q.put(("child", worker_id, idx, child.pid))

        # HARD wall-clock limit: this join can never hang forever.
        child.join(cfg["episode_timeout"])

        if child.is_alive():
            # Stuck: infinite loop in exec()'d code, dead JVM in env.step(),
            # or anything else. Kill the whole tree and move on.
            print(f"[worker {worker_id}] ep {idx} attempt {attempt} TIMEOUT after "
                  f"{cfg['episode_timeout']}s — killing process tree (pid {child.pid})", flush=True)
            _kill_tree(child.pid)
            child.join(60)
            if child.is_alive():
                try:
                    child.kill()
                except Exception:
                    pass
                child.join(10)
            if _result_path(cfg, idx).exists():
                status_q.put(("done", worker_id, idx, attempt))
            else:
                status_q.put(("failed", worker_id, idx, attempt, "timeout"))
        else:
            if child.exitcode == 0 and _result_path(cfg, idx).exists():
                status_q.put(("done", worker_id, idx, attempt))
            else:
                status_q.put(("failed", worker_id, idx, attempt, f"exit={child.exitcode}"))


def _spawn_worker(ctx, worker_id, task_q, status_q, cfg):
    p = ctx.Process(
        target=_worker_loop, args=(worker_id, task_q, status_q, cfg),
        daemon=False, name=f"swv-worker-{worker_id}",
    )
    p.start()
    return p


# ── Error result (written only after all attempts are exhausted) ──

def _write_error_result(cfg, idx, kind):
    task_name, var_idx = _indices()[idx]
    result = {
        "query": "",
        "name": task_name,
        "variation_idx": var_idx,
        "task_done": False,
        "reward": 0,
        "steps": 0,
        "messages": [],
        "relevant_skill_names": _safe_selection_names(idx),
        "error": f"episode failed after {cfg['max_attempts']} attempts: {kind}",
    }
    save_file = os.path.join(cfg["output_path"], f"idx_{idx}.json")
    tmp_file = f"{save_file}.tmp.{os.getpid()}"
    with open(tmp_file, "w") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    os.replace(tmp_file, save_file)
    print(f"[supervisor] ep {idx}: all attempts exhausted ({kind}) — recorded error result", flush=True)


def _safe_selection_names(idx):
    try:
        return _selections().get(str(idx), {}).get("relevant_skill_names", [])
    except Exception:
        return []


# ── Supervisor: shared queue + dynamic scheduling + no infinite waits ──

def _check_servers(base_port, num_ports, model):
    import urllib.request
    healthy = []
    wrong = []
    for p in range(base_port, base_port + num_ports):
        try:
            with urllib.request.urlopen(f"http://localhost:{p}/health", timeout=5) as r:
                if r.status != 200:
                    continue
        except Exception:
            continue
        # healthy — but is it actually serving OUR backbone?
        try:
            with urllib.request.urlopen(f"http://localhost:{p}/v1/models", timeout=5) as r:
                served = json.load(r)
            ids = [d.get("id") for d in served.get("data", [])]
        except Exception:
            ids = []
        if model in ids:
            healthy.append(p)
        else:
            wrong.append((p, ids))
    if wrong:
        for p, ids in wrong:
            print(f"[FATAL] port {p} is occupied by a DIFFERENT model: {ids}", flush=True)
        print(f"[FATAL] Ports {base_port}..{base_port + num_ports - 1} must serve "
              f"{model}. Stop the other vLLM servers first.", flush=True)
        sys.exit(1)
    if not healthy:
        print(f"[FATAL] No healthy vLLM server on ports "
              f"{base_port}..{base_port + num_ports - 1}. Start them first.", flush=True)
        sys.exit(1)
    if len(healthy) < num_ports:
        print(f"[WARN] Only {len(healthy)}/{num_ports} vLLM servers healthy: {healthy}. "
              f"Continuing (requests round-robin over all ports and retry).", flush=True)
    else:
        print(f"[OK] All {num_ports} online vLLM servers healthy and serving {model}: "
              f"ports {base_port}..{base_port + num_ports - 1}", flush=True)


def _supervise(args, cfg, tasks_to_run):
    ctx = multiprocessing.get_context("spawn")
    task_q = ctx.Queue()
    status_q = ctx.Queue()

    for idx in tasks_to_run:
        task_q.put((idx, 1))

    workers = [
        _spawn_worker(ctx, wid, task_q, status_q, cfg)
        for wid in range(args.max_workers)
    ]
    print(f"[supervisor] {args.max_workers} workers started on the shared queue; "
          f"{len(tasks_to_run)} episodes queued; episode_timeout={cfg['episode_timeout']}s; "
          f"max_attempts={cfg['max_attempts']}", flush=True)

    remaining = set(tasks_to_run)
    inflight = {}          # idx -> {wid, attempt, t, child_pid}
    retries_issued = 0
    errors_recorded = 0
    last_log = time.time()

    pbar = tqdm(total=len(tasks_to_run), desc=f"ScienceWorld-visskill-ctrl-qwen3.8max(ports={args.base_port}-{args.base_port + args.num_ports - 1})")

    def requeue_or_giveup(idx, attempt, kind):
        nonlocal retries_issued, errors_recorded
        if attempt < cfg["max_attempts"]:
            retries_issued += 1
            task_q.put((idx, attempt + 1))
            print(f"[supervisor] ep {idx}: {kind} on attempt {attempt} -> requeued "
                  f"(attempt {attempt + 1}/{cfg['max_attempts']})", flush=True)
        else:
            errors_recorded += 1
            _write_error_result(cfg, idx, kind)
            remaining.discard(idx)
            pbar.update(1)

    try:
        while remaining:
            # 1) drain status messages (never blocks: timeout=0.2s)
            while True:
                try:
                    msg = status_q.get(timeout=0.2)
                except _queue_mod.Empty:
                    break
                kind_msg = msg[0]
                if kind_msg == "took":
                    _, wid, idx, attempt = msg
                    inflight[idx] = {"wid": wid, "attempt": attempt, "t": time.time(), "child_pid": None}
                elif kind_msg == "child":
                    _, wid, idx, child_pid = msg
                    if idx in inflight:
                        inflight[idx]["child_pid"] = child_pid
                elif kind_msg == "done":
                    _, wid, idx, attempt = msg
                    inflight.pop(idx, None)
                    if idx in remaining:
                        remaining.discard(idx)
                        pbar.update(1)
                elif kind_msg == "failed":
                    _, wid, idx, attempt, kind = msg
                    inflight.pop(idx, None)
                    if _result_path(cfg, idx).exists():
                        # finished despite the reported failure (race) — count it
                        remaining.discard(idx)
                        pbar.update(1)
                    elif idx in remaining:
                        requeue_or_giveup(idx, attempt, kind)

            # 2) worker liveness: respawn dead workers, recover their episodes
            for wid in range(len(workers)):
                proc = workers[wid]
                if proc.is_alive():
                    continue
                print(f"[supervisor] worker {wid} died (exit={proc.exitcode}); recovering", flush=True)
                lost = [idx for idx, st in inflight.items() if st["wid"] == wid]
                for idx in lost:
                    st = inflight.pop(idx)
                    if st["child_pid"]:
                        _kill_tree(st["child_pid"])   # orphaned episode tree
                    if _result_path(cfg, idx).exists():
                        remaining.discard(idx)
                        pbar.update(1)
                    else:
                        requeue_or_giveup(idx, st["attempt"], f"worker-died({proc.exitcode})")
                workers[wid] = _spawn_worker(ctx, wid, task_q, status_q, cfg)

            # 3) deadline sweep: an in-flight episode far past its timeout
            #    means its worker is itself stuck — force-recover it.
            now = time.time()
            hard_deadline = cfg["episode_timeout"] + 180
            for idx, st in list(inflight.items()):
                if now - st["t"] > hard_deadline:
                    print(f"[supervisor] ep {idx} in flight for {int(now - st['t'])}s "
                          f"(> {hard_deadline}s); force-recovering", flush=True)
                    if st["child_pid"]:
                        _kill_tree(st["child_pid"])
                    inflight.pop(idx, None)
                    try:
                        workers[st["wid"]].terminate()
                    except Exception:
                        pass
                    if _result_path(cfg, idx).exists():
                        remaining.discard(idx)
                        pbar.update(1)
                    else:
                        requeue_or_giveup(idx, st["attempt"], "hard-deadline")

            # 4) periodic heartbeat for the log file
            if now - last_log > 120:
                last_log = now
                print(f"[heartbeat] remaining={len(remaining)} inflight={len(inflight)} "
                      f"retries={retries_issued} errors={errors_recorded}", flush=True)

            time.sleep(0.3)
    finally:
        pbar.close()
        # shut down workers
        try:
            for _ in workers:
                task_q.put(None)
        except Exception:
            pass
        deadline = time.time() + 60
        for proc in workers:
            proc.join(timeout=max(0.1, deadline - time.time()))
        for proc in workers:
            if proc.is_alive():
                try:
                    proc.terminate()
                except Exception:
                    pass
        # kill any episode trees still alive (e.g. interrupted run)
        for st in inflight.values():
            if st.get("child_pid"):
                _kill_tree(st["child_pid"])

    return retries_issued, errors_recorded


# ── Task categories & success metric ──
# Per-category summary
# (same mapping, same definitions: success = reward >= 100; by_category is
# grouped by longest task-name prefix), so summary.json is complete in one
# pass and no external post-processing script is needed.

CATEGORY_MAP = {
    'task-1': 'State',
    'task-2': 'Elec.',
    'task-2a': 'Elec.',
    'task-3': 'Class.',
    'task-4': 'Growth',
    'task-5': 'Chem.',
    'task-6': 'Life',
    'task-7': 'Stages',
    'task-10': 'Meas.',
}

CATEGORY_ORDER = ['State', 'Elec.', 'Class.', 'Growth', 'Chem.', 'Life', 'Stages', 'Meas.']


def get_category(task_name):
    """Longest-prefix category mapping used for the per-category summary."""
    for prefix in sorted(CATEGORY_MAP.keys(), key=len, reverse=True):
        if task_name.startswith(prefix):
            return CATEGORY_MAP[prefix]
    return 'Unknown'


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
    success_count = 0
    by_task = {}
    by_cat = {}

    for r in results:
        score = float(r.get("reward", 0) or 0)
        steps = int(r.get("steps", 0) or 0)
        done = bool(r.get("task_done", False))
        success = score >= 100
        task = r.get("name", "unknown")
        cat = get_category(task)

        score_sum += score
        steps_sum += steps
        if done:
            done_count += 1
        if success:
            success_count += 1

        d = by_task.setdefault(task, {"count": 0, "score_sum": 0.0, "steps_sum": 0,
                                      "done": 0, "success": 0})
        d["count"] += 1
        d["score_sum"] += score
        d["steps_sum"] += steps
        if done:
            d["done"] += 1
        if success:
            d["success"] += 1

        c = by_cat.setdefault(cat, {"count": 0, "score_sum": 0.0, "steps_sum": 0, "success": 0})
        c["count"] += 1
        c["score_sum"] += score
        c["steps_sum"] += steps
        if success:
            c["success"] += 1

    summary = {
        "experiment": "vis-skill-controlled-qwen3.8max-27B-scienceworld",
        "description": ("Controlled visual skill cards (same per-episode skills as the controlled "
                        "text-skill run — qwen3.8-max pre-annotation top-5 — rendered SKILL.md "
                        "via renderer v6; hang-proof dynamic scheduling) with Qwen3.5-27B on ScienceWorld"),
        "skill_selection": "pre-annotated by qwen3.8-max, top-5 relevance per episode "
                           "(science_skill_annotations.json)",
        "model": model,
        "benchmark": "ScienceWorld (SkillNet test split: 211 episodes, 24 tasks)",
        "max_steps": max_steps,
        "overall": {
            "total": total,
            "avg_score": round(score_sum / total, 2) if total else 0,
            "avg_steps": round(steps_sum / total, 1) if total else 0,
            "task_done": done_count,
            "task_done_rate": round(done_count / total * 100, 2) if total else 0,
            "success_count": success_count,
            "success_rate": round(success_count / total * 100, 2) if total else 0,
        },
        "by_task": {},
        "by_category": {},
    }

    for t in sorted(by_task.keys()):
        d = by_task[t]
        summary["by_task"][t] = {
            "count": d["count"],
            "avg_score": round(d["score_sum"] / d["count"], 2) if d["count"] else 0,
            "avg_steps": round(d["steps_sum"] / d["count"], 1) if d["count"] else 0,
            "task_done": d["done"],
            "success_count": d["success"],
            "success_rate": round(d["success"] / d["count"] * 100, 2) if d["count"] else 0,
        }

    for cat in CATEGORY_ORDER + (["Unknown"] if "Unknown" in by_cat else []):
        if cat not in by_cat:
            continue
        c = by_cat[cat]
        summary["by_category"][cat] = {
            "count": c["count"],
            "avg_score": round(c["score_sum"] / c["count"], 2) if c["count"] else 0,
            "success_count": c["success"],
            "success_rate": round(c["success"] / c["count"] * 100, 2) if c["count"] else 0,
            "avg_steps": round(c["steps_sum"] / c["count"], 2) if c["count"] else 0,
        }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"SUMMARY: {summary['experiment']}")
    print(f"{'='*60}")
    print(f"Overall: avg score {summary['overall']['avg_score']}, "
          f"avg {summary['overall']['avg_steps']} steps, "
          f"done {done_count}/{total} ({summary['overall']['task_done_rate']}%), "
          f"success(reward>=100) {success_count}/{total} "
          f"({summary['overall']['success_rate']}%)")
    print(f"\nPer task:")
    for t, d in summary["by_task"].items():
        print(f"  {t:<55} n={d['count']:>2}  score={d['avg_score']:>6.2f}  "
              f"steps={d['avg_steps']:>5.1f}  done={d['task_done']}/{d['count']}  "
              f"succ={d['success_count']}/{d['count']} ({d['success_rate']}%)")
    print(f"\nPer category:")
    for cat, d in summary["by_category"].items():
        print(f"  {cat:<8} n={d['count']:>3}  score={d['avg_score']:>6.2f}  "
              f"succ={d['success_count']}/{d['count']} ({d['success_rate']}%)  "
              f"steps={d['avg_steps']:>5.1f}")
    print(f"\nSaved to {summary_path}")


# ── Main ──

def main(args):
    indices = _indices()
    num_episodes = len(indices)

    # Load the controlled selections (qwen3.8-max annotation top-5). They are
    # validated by prepare_skill_selections.py to cover ALL episodes; any gap
    # here is a hard error because the episode would otherwise run skill-less.
    selections = _selections()
    missing_sel = [i for i in range(num_episodes) if str(i) not in selections]
    if missing_sel:
        print(f"[FATAL] skill_selections.json missing {len(missing_sel)} episodes: "
              f"{missing_sel[:20]}. Re-run code/prepare_skill_selections.py.")
        sys.exit(1)

    model_name = args.model.rstrip("/").split("/")[-1]
    output_path = f"{PROJECT_ROOT}/results/scienceworld/{model_name}/vis_skill"
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

    skip_set = existing | set(missing_sel)
    tasks_to_run = [i for i in range(ep_start, ep_end) if i not in skip_set]
    print(f"Total episodes: {num_episodes}, range [{ep_start}, {ep_end}), "
          f"finished={finished}, remaining={len(tasks_to_run)}, "
          f"vLLM ports={args.base_port}-{args.base_port + args.num_ports - 1}")

    _check_servers(args.base_port, args.num_ports, args.model)

    if tasks_to_run:
        cfg = {
            "model": args.model,
            "use_skill": args.use_skill,
            "max_steps": args.max_steps,
            "output_path": output_path,
            "base_port": args.base_port,
            "num_ports": args.num_ports,
            "client_timeout": args.client_timeout,
            "episode_timeout": args.episode_timeout,
            "max_attempts": args.max_attempts,
            "skill_cards_dir": str(SKILL_CARDS_DIR),
            "overall_procedure_examples_path": str(OVERALL_PROCEDURE_EXAMPLES_PATH),
            "procedure_code_template_path": str(PROCEDURE_CODE_TEMPLATE_PATH),
        }

        print(f"Starting {args.max_workers} workers on ONE shared queue "
              f"(true dynamic scheduling across {args.num_ports} online vLLM servers) ...")
        retries_issued, errors_recorded = _supervise(args, cfg, tasks_to_run)
        print(f"[done] retries issued: {retries_issued}, episodes recorded as errors: {errors_recorded}")

    # recount from disk
    finished = 0
    all_rewards = 0.0
    all_steps = 0
    for file in os.listdir(output_path):
        if file.endswith(".json") and file.startswith("idx_"):
            try:
                with open(os.path.join(output_path, file)) as f:
                    res = json.load(f)
                all_rewards += float(res.get("reward", 0) or 0)
                all_steps += int(res.get("steps", 0) or 0)
                finished += 1
            except Exception:
                continue

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
    else:
        missing = [i for i in range(ep_start, ep_end)
                   if not (Path(output_path) / f"idx_{i}.json").exists()]
        print(f"[WARN] {len(missing)} episodes missing after run: {missing[:50]}")


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max_workers", type=int, default=32)
    parser.add_argument("--max_steps", type=int, default=30)
    parser.add_argument("--use_skill", action="store_true")
    parser.add_argument("--base_port", type=int, default=8000)
    parser.add_argument("--num_ports", type=int, default=2)
    parser.add_argument("--client_timeout", type=float, default=300.0,
                        help="per-LLM-request HTTP timeout (seconds)")
    parser.add_argument("--episode_timeout", type=float, default=2400.0,
                        help="hard wall-clock limit per episode (seconds)")
    parser.add_argument("--max_attempts", type=int, default=2,
                        help="attempts per episode before recording an error result")
    parser.add_argument("--ep_start", type=int, default=-1)
    parser.add_argument("--ep_end", type=int, default=-1)
    args = parser.parse_args()
    main(args)
