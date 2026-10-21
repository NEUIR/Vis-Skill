"""
ScienceWorld BASELINE — NO SKILL (execution model: kimi-k2.5 via the
DashScope API).

API adaptation of scienceworld/gemma-4-31b/shared/code/run_scienceworld_baseline.py
(experiment flow kept EXACTLY the same; only the model endpoint changed):
the agent receives only the system prompt + task description, then acts step
by step (Thought/Action loop) — the model's base capability without any skill.

Endpoint (same pattern as alfworld/kimi-k2.5/baseline-noskill/code/run_noskill.py):
  * model: kimi-k2.5 (configurable via --model)
  * DashScope OpenAI-compatible API
      base_url: $DASHSCOPE_BASE_URL
                (default https://dashscope.aliyuncs.com/compatible-mode/v1)
      api_key : $DASHSCOPE_API_KEY
  * NO local vLLM servers / no port round-robin: all workers share the one
    API endpoint; each worker process lazily creates its own client.

Protocol follows the official SkillNet ScienceWorld evaluation:
  - SkillNet test split: shared/test_indices.json (symlink; 211 episodes
    over 24 tasks), env: scienceworld.ScienceWorldEnv,
    simplificationStr="easy", envStepLimit = max_steps,
    score = running max of info['score'] (0-100), temperature=0
    (request side; configurable via --temperature, negative = omit).

Scheduling is fully DYNAMIC: ALL episodes are submitted to ONE shared queue;
every persistent worker process, once it finishes an episode, pulls the NEXT
pending episode from that common queue (no static sharding — the run stays
busy until the very last task).

Robustness (unchanged from the gemma runners):
  - No unbounded waits. Every episode runs in its own grandchild process
    joined with a HARD wall-clock limit (--episode_timeout); on timeout the
    whole process group (python + ScienceWorld JVM) is SIGKILLed, the
    episode retried (--max_attempts) and, if still failing, recorded as an
    error result. Workers always return to the shared queue.
"""
import os
import re
import sys
import json
import time
import queue as _queue_mod
import signal
import argparse
import traceback
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

# ── Globals (set inside each episode process) ──

DEFAULT_MODEL = "kimi-k2.5"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

CODE_DIR = Path(__file__).resolve().parent           # scienceworld/kimi-k2.5/shared/code
SHARED_DIR = CODE_DIR.parent                         # scienceworld/kimi-k2.5/shared
# Shared dataset (symlink to the canonical copy — no duplication).
INDICES_PATH = SHARED_DIR / "test_indices.json"

_CLIENT_TIMEOUT = 300.0

_client = None
_TEMPERATURE = 0.0
_indices_cache = None


def _indices():
    global _indices_cache
    if _indices_cache is None:
        with open(INDICES_PATH) as f:
            _indices_cache = json.load(f)
    return _indices_cache


# ── Endpoint management (DashScope API, per-process client) ──

def _get_client():
    """One shared API endpoint; each worker/episode process lazily creates
    its own client (never created before a spawn/fork)."""
    global _client
    if _client is None:
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "DASHSCOPE_API_KEY is not set. Export your own DashScope "
                "API key first, e.g.:  export DASHSCOPE_API_KEY=<your-key>"
            )
        base_url = os.environ.get("DASHSCOPE_BASE_URL", DEFAULT_BASE_URL)
        _client = OpenAI(api_key=api_key, base_url=base_url, timeout=_CLIENT_TIMEOUT)
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
        raise ValueError(f"prompt must be a list or a string, got {type(prompt)}")
    response = _get_client().chat.completions.create(
        model=model,
        messages=messages,
        **_sampling_kwargs(),
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


# ── Core execution (no skill: system prompt + task description only) ──

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

    global _TEMPERATURE, _CLIENT_TIMEOUT
    _TEMPERATURE = cfg["temperature"]
    _CLIENT_TIMEOUT = cfg["client_timeout"]

    task_name, var_idx = "?", -1
    env = None
    try:
        task_name, var_idx = _indices()[idx]
        print(f"[ep {idx} att {attempt} wid {worker_id}] start {task_name} var {var_idx}", flush=True)

        env = ScienceWorldEnv(taskName=None, serverPath=None, envStepLimit=cfg["max_steps"])
        env.load(task_name, var_idx, simplificationStr="easy")

        result = scienceworld_run_single(
            env, task_name, var_idx, cfg["max_steps"], cfg["model"]
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
            # Stuck: dead JVM in env.step(), LLM stall, anything else.
            # Kill the whole tree and move on.
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
        daemon=False, name=f"sw-worker-{worker_id}",
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
        "relevant_skill_names": [],
        "error": f"episode failed after {cfg['max_attempts']} attempts: {kind}",
    }
    save_file = os.path.join(cfg["output_path"], f"idx_{idx}.json")
    tmp_file = f"{save_file}.tmp.{os.getpid()}"
    with open(tmp_file, "w") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    os.replace(tmp_file, save_file)
    print(f"[supervisor] ep {idx}: all attempts exhausted ({kind}) — recorded error result", flush=True)


# ── Supervisor: shared queue + dynamic scheduling + no infinite waits ──

def _check_endpoint(model):
    """Fail fast (before spawning any worker) if the API key is missing."""
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        print("[FATAL] DASHSCOPE_API_KEY is not set. Export your own DashScope "
              "API key first, e.g.:  export DASHSCOPE_API_KEY=<your-key>", flush=True)
        sys.exit(1)
    base_url = os.environ.get("DASHSCOPE_BASE_URL", DEFAULT_BASE_URL)
    print(f"[OK] DashScope API endpoint: {base_url}  model: {model}", flush=True)


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

    exp_tag = getattr(args, "experiment_name", "baseline-noskill")
    pbar = tqdm(total=len(tasks_to_run), desc=f"{exp_tag}(DashScope API)")

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

def generate_summary(output_path, summary_path, model, max_steps,
                     experiment_name="baseline-noskill-scienceworld"):
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

    model_name = model.rstrip("/").split("/")[-1]
    summary = {
        "experiment": experiment_name,
        "description": ("No-skill baseline (system prompt + task description only, direct "
                        "Thought/Action execution; hang-proof dynamic scheduling); "
                        f"executed by {model_name} via the DashScope API on ScienceWorld"),
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

    output_path = args.output_path   # per-experiment results dir, passed by launch.sh
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
          f"finished={finished}, remaining={len(tasks_to_run)} (DashScope API, "
          f"model={args.model})")

    _check_endpoint(args.model)

    if tasks_to_run:
        cfg = {
            "model": args.model,
            "max_steps": args.max_steps,
            "output_path": output_path,
            "client_timeout": args.client_timeout,
            "temperature": None if args.temperature < 0 else args.temperature,
            "episode_timeout": args.episode_timeout,
            "max_attempts": args.max_attempts,
        }

        print(f"Starting {args.max_workers} workers on ONE shared queue "
              f"(true dynamic scheduling against the DashScope API) ...")
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
        generate_summary(output_path, summary_path, args.model, args.max_steps,
                         experiment_name=args.experiment_name)
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
    parser.add_argument("--output_path", required=True,
                        help="per-experiment results directory (written by launch.sh)")
    parser.add_argument("--experiment_name", default="baseline-noskill-scienceworld",
                        help="experiment tag used in summary.json / progress bars")
    parser.add_argument("--max_workers", type=int, default=20,
                        help="number of dynamic worker processes pulling from ONE "
                             "shared queue (keep moderate to respect API rate limits)")
    parser.add_argument("--max_steps", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="sampling temperature passed to the API "
                             "(SkillNet protocol: 0; pass a negative value to "
                             "omit temperature entirely)")
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
