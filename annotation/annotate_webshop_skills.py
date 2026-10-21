#!/usr/bin/env python3
"""
Rank all 23 skills by relevance for each of the 200 WebShop test instructions
with Qwen3.8-Max (DashScope).
Output: per instruction, the ranking of every skill (most to least relevant)
with a relevance score and a reason.
"""

import argparse
import os
import json
import time
from pathlib import Path
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# ==================== Configuration ====================
API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL = "qwen3.8-max"

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inputs", "webshop_instances")
SKILLS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skills", "webshop")
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "webshop_skill_annotations.json")

CONCURRENCY = 8
MAX_RETRIES = 3
# ===============================================


def load_instructions(results_dir: str, count: int = 200) -> list:
    instructions = []
    for i in range(count):
        fp = os.path.join(results_dir, f"idx_{i}.json")
        if os.path.exists(fp):
            with open(fp, "r", encoding="utf-8") as f:
                d = json.load(f)
            query = d.get("query", "")
            if query.startswith("Instruction: "):
                query = query[len("Instruction: "):]
            instructions.append({"index": i, "instruction": query})
        else:
            instructions.append({"index": i, "instruction": ""})
    return instructions


def load_skills(skills_dir: str) -> dict:
    skills = {}
    for skill_path in sorted(Path(skills_dir).iterdir()):
        if not skill_path.is_dir():
            continue
        skill_md = skill_path / "SKILL.md"
        if not skill_md.exists():
            continue
        content = skill_md.read_text(encoding="utf-8")
        name = skill_path.name
        description = ""
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                frontmatter = parts[1]
                lines = frontmatter.split("\n")
                capture = False
                desc_lines = []
                for line in lines:
                    if line.strip().startswith("description:"):
                        rest = line.split("description:", 1)[1].strip()
                        if rest and rest not in (">-", ">", "|", ">'"):
                            description = rest.strip("'\"")
                            break
                        capture = True
                        continue
                    if capture:
                        if line.startswith("  ") or line.startswith("\t"):
                            desc_lines.append(line.strip())
                        else:
                            break
                if desc_lines and not description:
                    description = " ".join(desc_lines)
        skills[name] = description
    return skills


def build_prompt(skills: dict, instruction: str) -> str:
    skill_list = "\n".join(
        f"  {i+1}. {name}: {desc}"
        for i, (name, desc) in enumerate(skills.items())
    )

    prompt = f"""你是一个 AI Agent 技能规划专家。下面给出一个网购场景的用户指令，以及 23 个可用的 skill（技能）列表。

请你根据这条购物指令，对全部 23 个 skill 按照与该任务的相关性从高到低进行排序，并给出相关性评分（1-10分，10分最相关）。

## 可用 Skill 列表:
{skill_list}

## 用户购物指令:
{instruction}

## 要求:
1. 对全部 23 个 skill 按相关性从高到低排序
2. 给每个 skill 一个 1-10 的相关性评分（10=完成该任务必不可少，1=完全无关）
3. 对每个 skill 简要说明与该任务的关系（一句话）
4. 严格以如下 JSON 格式输出，不要输出其他内容:

```json
{{
  "skill_ranking": [
    {{
      "rank": 1,
      "skill_name": "最相关的skill名称",
      "relevance_score": 10,
      "reason": "与该任务的关系说明"
    }},
    {{
      "rank": 2,
      "skill_name": "第二相关的skill名称",
      "relevance_score": 9,
      "reason": "与该任务的关系说明"
    }},
    ...
  ]
}}
```"""
    return prompt


def call_qwen(client: OpenAI, prompt: str) -> dict:
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "你是一个精通 AI Agent 工作流规划的专家，擅长根据任务需求评估技能相关性并排序。请严格按要求的 JSON 格式输出，必须包含全部 23 个 skill。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=4096,
            )
            content = response.choices[0].message.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            result = json.loads(content)
            # Check that all 23 skills are present
            if "skill_ranking" in result and len(result["skill_ranking"]) >= 21:
                return result
            else:
                print(f"  [WARNING] too few skills returned: {len(result.get('skill_ranking', []))}, retrying...")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2)
                    continue
                return result
        except json.JSONDecodeError:
            print(f"  [WARNING] JSON parse failed (attempt {attempt+1}), raw response: {content[:300]}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2)
        except Exception as e:
            print(f"  [ERROR] API call failed (attempt {attempt+1}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(3)
    return None


def main():
    global OUTPUT_PATH
    parser = argparse.ArgumentParser(description="Rank WebShop skills by relevance with Qwen3.8-Max")
    parser.add_argument("--output", type=str, default=OUTPUT_PATH,
                        help="output JSON path; an existing file is resumed")
    OUTPUT_PATH = parser.parse_args().output

    if not API_KEY:
        print("Error: API key not set. Use your own DashScope API key: export DASHSCOPE_API_KEY=<your-key>")
        return

    print(f"Loading 200 instructions: {RESULTS_DIR}")
    instructions = load_instructions(RESULTS_DIR)
    valid_instructions = [x for x in instructions if x["instruction"]]
    print(f"Valid instructions: {len(valid_instructions)}")

    print(f"\nLoading skills: {SKILLS_DIR}")
    skills = load_skills(SKILLS_DIR)
    print(f"Loaded {len(skills)} skills:")
    for name in skills:
        print(f"  - {name}")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    # Resume from existing results
    results = []
    done_indices = set()
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            results = json.load(f)
        done_indices = {r["index"] for r in results}
        print(f"\n{len(results)} results already exist, skipping them...")

    tasks = [x for x in valid_instructions if x["index"] not in done_indices]
    print(f"\nAnnotating {len(tasks)} instructions (concurrency={CONCURRENCY})...\n")

    if not tasks:
        print("All tasks already done!")
        return

    lock = threading.Lock()
    completed_count = [0]

    def process_one(item):
        idx = item["index"]
        instruction = item["instruction"]
        prompt = build_prompt(skills, instruction)
        result = call_qwen(client, prompt)
        record = {
            "index": idx,
            "instruction": instruction,
            "annotation": result,
        }
        with lock:
            completed_count[0] += 1
            print(f"[{completed_count[0]}/{len(tasks)}] idx={idx}: {instruction[:60]}...")
        return record

    new_results = []
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        futures = {executor.submit(process_one, item): item for item in tasks}
        for future in as_completed(futures):
            record = future.result()
            new_results.append(record)

            if completed_count[0] % 20 == 0:
                all_results = results + sorted(new_results, key=lambda x: x["index"])
                with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                    json.dump(all_results, f, ensure_ascii=False, indent=2)
                print(f"  -> checkpoint: {len(all_results)} saved")

    # Final save
    results += sorted(new_results, key=lambda x: x["index"])
    results.sort(key=lambda x: x["index"])
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nDone! Annotated {len(results)} instructions, saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
