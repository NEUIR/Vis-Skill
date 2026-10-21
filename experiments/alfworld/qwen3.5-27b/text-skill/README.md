# ALFWorld / Qwen3.5-27B / text-skill

Controlled ALFWorld experiment: **text** skill presentation (SKILL.md only),
SkillNet 3-stage flow, executed by **Qwen3.5-27B**.

This is the backbone-swap twin of `../../qwen3.5-9b/text-skill`: code, prompts,
data and skills are identical; only the backbone-related parts differ
(model path, vLLM TP layout, ports).

Skill selection is **NOT done by the model at runtime**. It is fixed to the
**top-5 skills per instance** from the qwen3.8-max annotation
`code/alfworld_skill_ranking_per_instance.json` (sorted by `relevance_score`).

## Layout

- `code/run_text.py` — evaluation script (annotation-based retrieval + dynamic scheduling)
- `code/alfworld_skill_ranking_per_instance.json` — top-5 skill ranking from `annotation/outputs/`
- `code/base_config.yaml`, `code/alfworld_overall_procedure_examples.txt`,
  `code/alfworld_procedure_code_template.py` — SkillNet prompt examples and code template
- `skills_alfworld/` — the 37 skills
- `start_vllm.sh` / `stop_vllm.sh` — online vLLM deployment
  (**2 replicas x TP=2**: GPU(0,1) -> port 8000, GPU(2,3) -> port 8001)
- `launch.sh` — runs the eval with 128 dynamic workers pulling from one shared queue
- `results/alfworld/Qwen3.5-27B/text_baseline/idx_*.json` — per-game results (written by `launch.sh`)

## Run

```bash
bash start_vllm.sh   # deploy 2 online vLLM servers (TP=2 each), wait until healthy
bash launch.sh       # evaluate 134 OOD games (resume-safe, skips existing idx_*.json)
bash stop_vllm.sh    # when done
```

Scheduling: every worker, once it finishes a game, pulls the next game from a
single shared queue and is round-robin assigned to a vLLM port at pick-up
time, so both replicas stay busy until the last task (true dynamic
scheduling, no static partitioning).

Environment: `envs/environment_alfworld.yml`
(vllm 0.26.0, alfworld 0.4.2). Data: `ALFWORLD_DATA=data/alfworld`.
