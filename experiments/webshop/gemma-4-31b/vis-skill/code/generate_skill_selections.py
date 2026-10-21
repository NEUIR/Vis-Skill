"""
Generate skill_selections.json from webshop_skill_annotations.json.

Selects the top-5 most relevant skills per instance based on Qwen3.8-Max
annotations (controlled selection — NO model-based skill retrieval).
Output format matches the skill_selections.json consumed by run_visual.py,
and the selections are IDENTICAL to what the text-skill experiment uses
(same annotation file, same top-K rule), so text-skill and vis-skill use
exactly the same skills per task by construction.
"""
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Canonical Qwen3.8-Max annotation file shared by all WebShop experiments.
ANNOTATIONS_PATH = Path(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "..", "annotation", "outputs", "webshop_skill_annotations.json")
)
TEST_INDICES_PATH = PROJECT_ROOT / "test_indices.json"
OUTPUT_PATH = PROJECT_ROOT / "skill_selections.json"

TOP_K = 5


def main():
    with open(ANNOTATIONS_PATH) as f:
        annotations = json.load(f)

    with open(TEST_INDICES_PATH) as f:
        session_ids = json.load(f)

    # Build index -> annotation mapping
    ann_by_index = {item["index"]: item for item in annotations}

    selections = {}
    for idx in range(len(session_ids)):
        session_id = session_ids[idx]
        ann = ann_by_index.get(idx)
        if ann is None:
            print(f"[WARN] No annotation for index {idx}, using empty skills")
            selections[str(idx)] = {
                "session_id": session_id,
                "query": "",
                "relevant_skill_names": [],
            }
            continue

        # Select top-K skills by rank (rank 1 = most relevant)
        ranking = ann["annotation"]["skill_ranking"]
        top_skills = [s["skill_name"] for s in sorted(ranking, key=lambda x: x["rank"])[:TOP_K]]

        selections[str(idx)] = {
            "session_id": session_id,
            "query": f"Instruction: {ann['instruction']}",
            "relevant_skill_names": top_skills,
        }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(selections, f, indent=2, ensure_ascii=False)

    print(f"Generated {OUTPUT_PATH}")
    print(f"  Total instances: {len(selections)}")
    print(f"  Top-K: {TOP_K}")
    # Show skill frequency
    from collections import Counter
    all_skills = []
    for v in selections.values():
        all_skills.extend(v["relevant_skill_names"])
    freq = Counter(all_skills).most_common(10)
    print(f"  Top skill frequencies:")
    for skill, count in freq:
        print(f"    {skill}: {count}")


if __name__ == "__main__":
    main()
