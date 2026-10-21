# ALFWorld / Qwen3.5-9B / text-skill

Controlled ALFWorld experiment: **text** skill presentation (SKILL.md only),
SkillNet 3-stage flow, executed by **Qwen3.5-9B**.

Skill selection is **NOT done by the model at runtime**. It is fixed to the
**top-5 skills per instance** from the qwen3.8-max annotation
`code/alfworld_skill_ranking_per_instance.json` (sorted by `relevance_score`).
Stages 2/3 (procedure text + procedure code) and the agent loop follow the
SkillNet pipeline unchanged.

## Layout

- `code/run_text.py` — evaluation script (annotation-based retrieval + dynamic scheduling)
- `code/alfworld_skill_ranking_per_instance.json` — top-5 skill ranking from `annotation/outputs/`
- `code/base_config.yaml`, `code/alfworld_overall_procedure_examples.txt`,
  `code/alfworld_procedure_code_template.py` — SkillNet prompt examples and code template
- `skills_alfworld/` — the 37 skills
- `start_vllm.sh` / `stop_vllm.sh` — online vLLM deployment (4 servers, GPU i -> port 8000+i, TP=1)
- `launch.sh` — runs the eval with 128 dynamic workers pulling from one shared queue
- `results/alfworld/Qwen3.5-9B/text_baseline/idx_*.json` — per-game results (written by `launch.sh`)

## Run

```bash
bash start_vllm.sh   # deploy 4 online vLLM servers, wait until healthy
bash launch.sh       # evaluate 134 OOD games (resume-safe, skips existing idx_*.json)
bash stop_vllm.sh    # when done
```

Scheduling: every worker, once it finishes a game, pulls the next game from a
single shared queue and is round-robin assigned to a vLLM port at pick-up
time, so all 4 GPUs stay busy until the last task (true dynamic scheduling,
no static partitioning).

Environment: `envs/environment_alfworld.yml`
(vllm 0.26.0, alfworld 0.4.2). Data: `ALFWORLD_DATA=data/alfworld`.
