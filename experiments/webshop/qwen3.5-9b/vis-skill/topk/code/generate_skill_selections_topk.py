"""
Generate skill_selections.json for the vis-skill top-k series from
webshop_skill_annotations.json.

Selects the top-K most relevant skills per instance based on Qwen3.8-Max
annotations (K set via --top_k; level0 baseline used 5:
level-2->3, level+2->7, level+4->9, level+8->13, level+16->21).

Annotations / test indices are read from the parent v1 directory
(vis-skill/webshop_skill_annotations.json);
output is written locally to vis-skill/topk/skill_selections.json.
"""
import json
import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent     # vis-skill/topk/code
PROJECT_ROOT = SCRIPT_DIR.parent                 # vis-skill/topk
PARENT = PROJECT_ROOT.parent                     # vis-skill/

ANNOTATIONS_PATH = PARENT / "webshop_skill_annotations.json"
TEST_INDICES_PATH = PARENT / "test_indices.json"
OUTPUT_PATH = PROJECT_ROOT / "skill_selections.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--top_k', type=int, default=5,
                        help="Number of skills to select per instance")
    args = parser.parse_args()
    top_k = args.top_k

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

        # Select top-K skills by rank
        ranking = ann["annotation"]["skill_ranking"]
        top_skills = [s["skill_name"] for s in sorted(ranking, key=lambda x: x["rank"])[:top_k]]

        selections[str(idx)] = {
            "session_id": session_id,
            "query": f"Instruction: {ann['instruction']}",
            "relevant_skill_names": top_skills,
        }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(selections, f, indent=2, ensure_ascii=False)

    print(f"Generated {OUTPUT_PATH}")
    print(f"  Total instances: {len(selections)}")
    print(f"  Top-K: {top_k}")
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
