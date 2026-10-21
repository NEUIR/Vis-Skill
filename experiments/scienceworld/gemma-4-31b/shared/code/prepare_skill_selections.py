"""
Build + verify the CONTROLLED per-task skill selections for the vis-skill
experiment from the qwen3.8-max pre-annotation.

Controlled-experiment replacement of the v1 script (which derived the
selections from the text-skill run results). Here the selections are FIXED
by science_skill_annotations.json (annotation/annotate_science_skills.py,
model qwen3.8-max): for every episode the TOP-5 most relevant skills by
ranked relevance are used, identically for the text-skill and the
vis-skill run.

Modes
-----
build (default):
    --annotations --indices --skills-dir --out
  1. Loads the annotation and test_indices.json.
  2. HARD GATE: requires one complete annotation entry per episode
     (task/variation consistent, >= TOP_K ranked skills, every selected
     skill present in the local skill library by frontmatter name).
     Exits non-zero on any problem — do not proceed to vis-skill.
  3. Writes ../skill_selections.json (consumed by run_scienceworld_vis.py)
     plus a human-readable report.

cross-check (adds --text-results):
  Additionally verifies that the finished controlled text-skill run used
  EXACTLY the annotation top-5 for every episode (relevant_skill_names in
  idx_*.json, order-sensitive) and back-fills the per-episode task query
  into skill_selections.json. Any missing/mismatched episode exits
  non-zero: a mismatch would mean the text run did not honor the
  controlled selections, which invalidates the text-vs-vis comparison.
"""
import json
import argparse
import sys
from pathlib import Path

import yaml

TOP_K = 5


def load_library_names(skills_dir):
    """Frontmatter names of the local skill library (same parsing as the runners)."""
    names = set()
    for skill_dir in sorted(Path(skills_dir).iterdir()):
        if not skill_dir.is_dir():
            continue
        md = skill_dir / "SKILL.md"
        if not md.exists():
            continue
        try:
            content = md.read_text(encoding="utf-8")
            if content.strip().startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    header = yaml.safe_load(parts[1])
                    if isinstance(header, dict) and header.get("name"):
                        names.add(header["name"])
        except Exception as e:
            print(f"[ERROR] parsing {skill_dir.name}/SKILL.md: {e}")
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", required=True,
                    help="science_skill_annotations.json (qwen3.8-max ranking)")
    ap.add_argument("--indices", required=True,
                    help="test_indices.json (SkillNet ScienceWorld test split)")
    ap.add_argument("--skills-dir", required=True,
                    help="local skills_scienceworld/ directory (frontmatter-name check)")
    ap.add_argument("--out", required=True,
                    help="output skill_selections.json for the vis-skill experiment")
    ap.add_argument("--text-results", default="",
                    help="optional: controlled text-skill results dir with idx_*.json "
                         "for the cross-check + query back-fill")
    args = ap.parse_args()

    with open(args.indices) as f:
        indices = json.load(f)
    with open(args.annotations) as f:
        ann = json.load(f)

    ann_by_idx = {}
    for entry in ann.get("task_skill_rankings", []):
        ann_by_idx[int(entry["instance_id"])] = entry
    print(f"[INFO] Annotation model: {ann.get('metadata', {}).get('model', '?')}, "
          f"{len(ann_by_idx)} annotated episodes, TOP_K={TOP_K}")

    library = load_library_names(args.skills_dir)
    print(f"[INFO] Skill library: {len(library)} skills ({args.skills_dir})")

    # ── Build selections from the annotation (hard gate) ──
    selections = {}
    problems = []
    skill_counter = {}
    count_dist = {}

    for idx, (task_name, var_idx) in enumerate(indices):
        entry = ann_by_idx.get(idx)
        if entry is None:
            problems.append(f"episode {idx}: no annotation entry")
            continue
        if entry.get("task_name") != task_name or int(entry.get("variation_idx", -1)) != int(var_idx):
            problems.append(
                f"episode {idx}: annotation says {entry.get('task_name')}/"
                f"var{entry.get('variation_idx')}, expected {task_name}/var{var_idx}"
            )
            continue
        if entry.get("error"):
            problems.append(f"episode {idx}: annotation recorded an error: {entry['error']}")
            continue
        ranked = entry.get("ranked_skills", [])
        names = [s["skill_name"] for s in ranked[:TOP_K]]
        if len(names) < TOP_K:
            problems.append(f"episode {idx}: only {len(names)} ranked skills (< {TOP_K})")
            continue
        unknown = [n for n in names if n not in library]
        if unknown:
            problems.append(f"episode {idx}: skills not in library: {unknown}")
            continue

        selections[str(idx)] = {
            "task_name": task_name,
            "variation_idx": var_idx,
            "query": "",   # back-filled from text results when available
            "relevant_skill_names": names,
            "relevance_scores": [s.get("relevance_score") for s in ranked[:TOP_K]],
        }
        k = len(names)
        count_dist[k] = count_dist.get(k, 0) + 1
        for n in names:
            skill_counter[n] = skill_counter.get(n, 0) + 1

    if problems:
        print(f"[ERROR] {len(problems)} annotation problems — NOT writing selections:")
        for p in problems[:30]:
            print(f"  {p}")
        sys.exit(1)

    # ── Optional cross-check against the controlled text-skill run ──
    if args.text_results:
        text_results = Path(args.text_results)
        if not text_results.exists():
            print(f"[ERROR] text-skill results dir not found: {text_results}")
            sys.exit(1)
        mismatches = []
        for idx, (task_name, var_idx) in enumerate(indices):
            f = text_results / f"idx_{idx}.json"
            if not f.exists():
                mismatches.append(f"episode {idx}: idx_{idx}.json missing in text results")
                continue
            with open(f) as fh:
                d = json.load(fh)
            if d.get("name") != task_name or int(d.get("variation_idx", -1)) != int(var_idx):
                mismatches.append(
                    f"episode {idx}: text result says {d.get('name')}/var{d.get('variation_idx')}, "
                    f"expected {task_name}/var{var_idx}"
                )
                continue
            used = d.get("relevant_skill_names", [])
            expected = selections[str(idx)]["relevant_skill_names"]
            if used != expected:
                mismatches.append(
                    f"episode {idx}: text run used {used}, annotation top-{TOP_K} is {expected}"
                )
                continue
            selections[str(idx)]["query"] = d.get("query", "")
        if mismatches:
            print(f"[ERROR] {len(mismatches)} text-vs-annotation mismatches — "
                  f"the controlled text-skill run did not honor the selections:")
            for m in mismatches[:30]:
                print(f"  {m}")
            sys.exit(1)
        print(f"[OK] Cross-check passed: all {len(indices)} text-skill episodes used "
              f"exactly the annotation top-{TOP_K}.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(selections, f, indent=2, ensure_ascii=False)

    report_path = out_path.with_name(out_path.stem + "_report.txt")
    with open(report_path, "w") as f:
        f.write("Per-episode CONTROLLED skill selections "
                f"(qwen3.8-max annotation, top-{TOP_K} relevance)\n")
        f.write("=" * 100 + "\n")
        for idx in range(len(indices)):
            s = selections[str(idx)]
            f.write(f"episode {idx:>3} | {s['task_name']} (var {s['variation_idx']}) | "
                    f"skills: {s['relevant_skill_names']}\n")
        f.write("\nSkill usage frequency:\n")
        for name, c in sorted(skill_counter.items(), key=lambda x: -x[1]):
            f.write(f"  {name}: {c}\n")
        f.write("\n#skills-per-episode distribution:\n")
        for k in sorted(count_dist):
            f.write(f"  {k} skills: {count_dist[k]} episodes\n")

    print(f"[OK] Verified {len(selections)}/{len(indices)} episodes "
          f"(annotation complete, task/variation consistent, all skills in library).")
    print(f"[OK] {len(skill_counter)} unique skills used; "
          f"#skills-per-episode distribution: { {k: count_dist[k] for k in sorted(count_dist)} }")
    print(f"[OK] Wrote {out_path}")
    print(f"[OK] Wrote {report_path}")


if __name__ == "__main__":
    main()
