"""
For every ALFWorld instance run by SkillNet, ask Qwen3.8-Max (DashScope) to rank all 37 skills by relevance.

Usage:
    export DASHSCOPE_API_KEY=<your own DashScope API key>
    python annotate_alfworld_skills.py

Output:
    alfworld_skill_ranking_per_instance.json
"""
import os
import json
import re
import time
import glob
from pathlib import Path
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ── Configuration ──
API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL = "qwen3.8-max"

SKILLS_DIR = Path(__file__).parent / "skills" / "alfworld"
RESULTS_DIR = Path(__file__).parent / "inputs" / "alfworld_instances"
OUTPUT_FILE = Path(__file__).parent / "outputs" / "alfworld_skill_ranking_per_instance.json"

MAX_WORKERS = 4  # concurrency; keep low to avoid API rate limits

# ── Initialize client ──
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def load_skills():
    """Load all skill names and descriptions from SKILL.md files."""
    skills = []
    for skill_dir in sorted(SKILLS_DIR.iterdir()):
        if not skill_dir.is_dir():
            continue
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            continue
        content = skill_md.read_text(encoding="utf-8")
        name_match = re.search(r"^name:\s*(.+)$", content, re.MULTILINE)
        desc_match = re.search(r"^description:\s*(.+)$", content, re.MULTILINE)
        if name_match and desc_match:
            skills.append({
                "name": name_match.group(1).strip(),
                "description": desc_match.group(1).strip(),
            })
    return skills


def load_instances():
    """Load all SkillNet result instances."""
    instances = []
    for fp in sorted(RESULTS_DIR.glob("idx_*.json"), key=lambda p: int(p.stem.split("_")[1])):
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        instances.append({
            "idx": int(fp.stem.split("_")[1]),
            "file": fp.name,
            "name": data.get("name", ""),
            "query": data.get("query", ""),
            "task_type": data.get("name", "").split("-")[0] + "-" + "-".join(data.get("name", "").split("-")[1:]).split("/")[0].rsplit("-", 1)[0] if data.get("name") else "",
        })
    return instances


def rank_skills_for_instance(instance, skills, max_retries=3):
    """Call Qwen3.8-Max to rank all skills by relevance for a specific instance."""
    skills_text = "\n".join(
        f"{i+1}. **{s['name']}**: {s['description']}"
        for i, s in enumerate(skills)
    )

    task_name = instance["name"].split("/")[0] if "/" in instance["name"] else instance["name"]
    task_type = task_name.split("-")[0] if "-" in task_name else task_name
    # Extract proper task type prefix
    for prefix in ["look_at_obj_in_light", "pick_and_place_simple", "pick_and_place_with_movable_recep",
                   "pick_clean_then_place_in_recep", "pick_cool_then_place_in_recep",
                   "pick_heat_then_place_in_recep", "pick_two_obj_and_place"]:
        if task_name.startswith(prefix):
            task_type = prefix
            break

    prompt = f"""You are an expert in embodied AI agent task decomposition for the ALFWorld household environment.

## Task Instance
- **Task type**: {task_type}
- **Task name**: {task_name}
- **Goal**: {instance["query"]}

## Available Skills ({len(skills)} total)
{skills_text}

## Your Job
Rank ALL {len(skills)} skills by their relevance to completing this specific task instance: "{instance["query"]}".
- Relevance score from 1 (not relevant at all) to 10 (essential/critical for this task).
- Consider which skills would be directly invoked during execution, which are supportive/auxiliary, and which are irrelevant.
- Return a JSON array sorted from MOST relevant (highest score) to LEAST relevant (lowest score).

## Output Format (strict JSON, no markdown fences, no extra text)
[
  {{"skill_name": "xxx", "relevance_score": 10, "reason": "brief reason"}},
  {{"skill_name": "xxx", "relevance_score": 8, "reason": "brief reason"}},
  ...
]

Return ONLY the JSON array."""

    messages = [{"role": "user", "content": prompt}]

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.1,
            )
            raw = response.choices[0].message.content.strip()
            # Strip thinking tags if present (qwen3 thinking mode)
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            # Strip markdown fences
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            ranking = json.loads(raw)
            if isinstance(ranking, list) and len(ranking) > 0:
                return ranking
        except (json.JSONDecodeError, Exception) as e:
            print(f"  [RETRY {attempt+1}/{max_retries}] idx_{instance['idx']}: {e}")
            time.sleep(3 * (attempt + 1))

    print(f"  [FAIL] idx_{instance['idx']}: Could not get valid ranking")
    return None


def process_single(args):
    """Worker function for thread pool."""
    instance, skills = args
    ranking = rank_skills_for_instance(instance, skills)
    return instance["idx"], ranking


def main():
    print("Loading skills...")
    skills = load_skills()
    print(f"  Found {len(skills)} skills")

    print("Loading SkillNet instances...")
    instances = load_instances()
    print(f"  Found {len(instances)} instances\n")

    # Check for existing partial results to support resume
    partial_results = {}
    if OUTPUT_FILE.exists():
        try:
            existing = json.loads(OUTPUT_FILE.read_text(encoding="utf-8"))
            if "instances" in existing:
                for item in existing["instances"]:
                    if item.get("skill_ranking"):
                        partial_results[item["idx"]] = item
                print(f"  Resuming: {len(partial_results)} instances already done\n")
        except Exception:
            pass

    # Filter out already-completed instances
    todo = [inst for inst in instances if inst["idx"] not in partial_results]
    print(f"  To process: {len(todo)} instances\n")

    results = list(partial_results.values())

    # Process with thread pool
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_single, (inst, skills)): inst
            for inst in todo
        }
        pbar = tqdm(total=len(todo), desc="Ranking skills")
        for future in as_completed(futures):
            inst = futures[future]
            try:
                idx, ranking = future.result()
                entry = {
                    "idx": idx,
                    "file": inst["file"],
                    "task_name": inst["name"],
                    "query": inst["query"],
                    "skill_ranking": ranking if ranking else [],
                }
                results.append(entry)
                if ranking:
                    top3 = [r["skill_name"] for r in ranking[:3]]
                    pbar.set_postfix_str(f"idx_{idx} top3={top3}")
            except Exception as e:
                print(f"  [ERROR] idx_{inst['idx']}: {e}")
                results.append({
                    "idx": inst["idx"],
                    "file": inst["file"],
                    "task_name": inst["name"],
                    "query": inst["query"],
                    "skill_ranking": [],
                    "error": str(e),
                })
            pbar.update(1)

            # Periodically save (every 10 completions)
            if len(results) % 10 == 0:
                _save(results, skills)

        pbar.close()

    # Final save
    _save(results, skills)
    print(f"\nDone! Results saved to: {OUTPUT_FILE}")


def _save(results, skills):
    """Save current results to file."""
    results_sorted = sorted(results, key=lambda x: x["idx"])
    output = {
        "metadata": {
            "model": MODEL,
            "total_skills": len(skills),
            "total_instances": len(results_sorted),
            "skills_list": [s["name"] for s in skills],
        },
        "instances": results_sorted,
    }
    OUTPUT_FILE.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
