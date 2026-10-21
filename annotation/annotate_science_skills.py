#!/usr/bin/env python3
"""
For every SkillNet ScienceWorld task, ask Qwen3.8-Max (DashScope) to rank
all skills by relevance; writes a JSON file.
"""

import os
import json
import time
import re
import glob
import argparse
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ============================================================
# Configuration
# ============================================================
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL = "qwen3.8-max"  # Qwen3.8-Max model ID on DashScope

SKILLS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skills", "scienceworld")
TASKNAME2ID_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inputs", "scienceworld", "taskname2id.json")
TASKS_JSON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inputs", "scienceworld", "tasks.json")
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "science_skill_annotations.json")


def load_skills(skills_dir):
    """Read the name and description of every skill."""
    skills = []
    skill_dirs = sorted(glob.glob(os.path.join(skills_dir, "*")))
    
    for skill_dir in skill_dirs:
        skill_md_path = os.path.join(skill_dir, "SKILL.md")
        if not os.path.isfile(skill_md_path):
            continue
        
        with open(skill_md_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Parse the frontmatter
        name = os.path.basename(skill_dir)
        description = ""
        
        # Extract name and description from the YAML frontmatter
        fm_match = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
        if fm_match:
            fm_text = fm_match.group(1)
            name_match = re.search(r"^name:\s*(.+)$", fm_text, re.MULTILINE)
            desc_match = re.search(r"^description:\s*(.+)$", fm_text, re.MULTILINE)
            if name_match:
                name = name_match.group(1).strip()
            if desc_match:
                description = desc_match.group(1).strip()
        
        # Instructions section (everything after the frontmatter)
        instructions = content[fm_match.end():].strip() if fm_match else content.strip()
        # Keep only the first 500 characters so the prompt stays short
        instructions_brief = instructions[:500]
        
        skills.append({
            "name": name,
            "description": description,
            "instructions_brief": instructions_brief
        })
    
    return skills


def load_tasks(taskname2id_path, tasks_json_path):
    """Load every task instance run by SkillNet (each entry of test_indices.json)."""
    # Load test_indices — each entry is [task_name, variation_idx]
    test_indices_path = os.path.join(os.path.dirname(taskname2id_path), "test_indices.json")
    with open(test_indices_path, "r") as f:
        test_indices = json.load(f)
    
    # Load the detailed task descriptions
    with open(tasks_json_path, "r") as f:
        tasks_detail = json.load(f)
    
    tasks = []
    for idx, (task_name, var_idx) in enumerate(test_indices):
        task_info = {
            "task_name": task_name,
            "variation_idx": var_idx,
            "instance_id": idx,
            "topic": "",
            "task_description": "",
        }
        # Match the readable description from tasks.json
        for t in tasks_detail:
            if t["task_name"] in task_name:
                task_info["topic"] = t.get("topic", "")
                task_info["task_description"] = t.get("task", "")
                break
        tasks.append(task_info)
    
    return tasks


def build_prompt(task, skills):
    """Build the ranking prompt for the LLM."""
    # Build the skill list text
    skill_list_text = ""
    for i, skill in enumerate(skills, 1):
        skill_list_text += f"{i}. **{skill['name']}**: {skill['description']}\n"
    
    prompt = f"""你是一个 AI agent 技能相关性评估专家。

## 任务
给定一个 ScienceWorld 环境中的任务，请对下面所有的技能（skill）按照与该任务的相关性从高到低进行排序。

## ScienceWorld 任务
- 任务名称: {task['task_name']}
- 变体编号: {task['variation_idx']}
- 实例 ID: {task['instance_id']}
- 主题领域: {task.get('topic', 'N/A')}
- 任务描述: {task.get('task_description', 'N/A')}

## 可用技能列表
{skill_list_text}

## 要求
1. 对所有 {len(skills)} 个技能按照与上述任务的相关性从高到低排序
2. 为每个技能给出一个 1-10 的相关性分数（10=最相关，1=完全不相关）
3. 简要说明相关性理由（一句话）

## 输出格式
严格按以下 JSON 格式输出，不要输出其他内容：
```json
{{
  "task_name": "{task['task_name']}",
  "variation_idx": {task['variation_idx']},
  "ranked_skills": [
    {{
      "rank": 1,
      "skill_name": "技能名称",
      "relevance_score": 10,
      "reason": "相关性理由"
    }},
    ...
  ]
}}
```

注意：必须包含所有 {len(skills)} 个技能，从最相关到最不相关排列。只输出 JSON，不要其他文字。"""
    
    return prompt


def call_qwen(client, prompt, max_retries=3):
    """Call the Qwen3.8-Max API (DashScope, OpenAI-compatible)."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=16384,
            )
            content = response.choices[0].message.content
            return content
        except Exception as e:
            print(f"  API call failed (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))
    return None


def parse_response(response_text):
    """Parse the JSON returned by the LLM."""
    if not response_text:
        return None
    
    # Try to extract a JSON block
    json_match = re.search(r"```json\s*(.*?)\s*```", response_text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # Try to parse the whole response directly
        json_str = response_text.strip()
        # Strip any leading/trailing text
        if json_str.startswith("```"):
            json_str = re.sub(r"^```\w*\n?", "", json_str)
            json_str = re.sub(r"\n?```$", "", json_str)
    
    try:
        result = json.loads(json_str)
        return result
    except json.JSONDecodeError as e:
        print(f"  JSON parse failed: {e}")
        print(f"  First 200 chars of raw response: {response_text[:200]}")
        return None


def process_single_task(client, task, skills):
    """Rank the skills for a single task."""
    prompt = build_prompt(task, skills)
    response = call_qwen(client, prompt)
    result = parse_response(response)
    
    if result is None:
        return {
            "task_name": task["task_name"],
            "variation_idx": task["variation_idx"],
            "instance_id": task["instance_id"],
            "ranked_skills": [],
            "error": "解析失败"
        }
    
    result["variation_idx"] = task["variation_idx"]
    result["instance_id"] = task["instance_id"]
    return result


def main():
    global MODEL
    
    parser = argparse.ArgumentParser(description="Rank skills by relevance for ScienceWorld tasks with Qwen3.8-Max")
    parser.add_argument("--model", type=str, default=MODEL, help="model name")
    parser.add_argument("--output", type=str, default=OUTPUT_PATH, help="output JSON path")
    parser.add_argument("--workers", type=int, default=3, help="number of parallel threads")
    parser.add_argument("--skills_dir", type=str, default=SKILLS_DIR, help="skills directory")
    parser.add_argument("--taskname2id", type=str, default=TASKNAME2ID_PATH, help="path to taskname2id.json")
    parser.add_argument("--tasks_json", type=str, default=TASKS_JSON_PATH, help="path to tasks.json")
    args = parser.parse_args()
    
    MODEL = args.model
    
    # 1. Get the API key
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        print("Error: API key not set. Use your own DashScope API key: export DASHSCOPE_API_KEY=<your-key>")
        return
    
    # 2. Initialise the client
    client = OpenAI(
        api_key=api_key,
        base_url=DASHSCOPE_BASE_URL,
    )
    
    # 3. Load skills
    print(f"Loading skills from: {args.skills_dir}")
    skills = load_skills(args.skills_dir)
    print(f"  Loaded {len(skills)} skills")
    
    # 4. Load tasks
    print(f"Loading tasks from: {args.taskname2id}")
    tasks = load_tasks(args.taskname2id, args.tasks_json)
    print(f"  {len(tasks)} tasks to rank")
    
    # 5. Rank every task
    print(f"\nRanking (model: {MODEL}, parallel: {args.workers} threads)...")
    print("=" * 60)
    
    results = []
    
    if args.workers == 1:
        # Sequential
        for task in tqdm(tasks, desc="ranking"):
            print(f"\nTask: {task['task_name']}")
            result = process_single_task(client, task, skills)
            results.append(result)
            skill_count = len(result.get("ranked_skills", []))
            print(f"  Done: {skill_count} skills ranked")
            time.sleep(1)  # avoid rate limits
    else:
        # Parallel
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_task = {
                executor.submit(process_single_task, client, task, skills): task
                for task in tasks
            }
            
            pbar = tqdm(total=len(tasks), desc="ranking")
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    results.append(result)
                    skill_count = len(result.get("ranked_skills", []))
                    pbar.set_postfix({"task": task["task_name"][:30], "skills": skill_count})
                except Exception as e:
                    print(f"\nTask {task['task_name']} var {task['variation_idx']} raised: {e}")
                    results.append({
                        "task_name": task["task_name"],
                        "variation_idx": task["variation_idx"],
                        "instance_id": task["instance_id"],
                        "ranked_skills": [],
                        "error": str(e)
                    })
                pbar.update(1)
            pbar.close()
    
    # 6. Sort results by instance_id
    results.sort(key=lambda x: x.get("instance_id", 999))
    
    # 7. Build the final output
    output = {
        "metadata": {
            "model": MODEL,
            "total_tasks": len(tasks),
            "total_skills": len(skills),
            "skill_names": [s["name"] for s in skills],
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "task_skill_rankings": results
    }
    
    # 8. Save
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'=' * 60}")
    print(f"Done! Results saved to: {args.output}")
    print(f"  - tasks: {len(results)}")
    print(f"  - skills: {len(skills)}")
    
    # Success rate
    success = sum(1 for r in results if len(r.get("ranked_skills", [])) > 0)
    print(f"  - succeeded: {success}/{len(results)}")


if __name__ == "__main__":
    main()
