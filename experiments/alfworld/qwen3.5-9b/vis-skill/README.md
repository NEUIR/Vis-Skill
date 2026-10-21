# ALFWorld / Qwen3.5-9B / vis-skill

Controlled ALFWorld experiment: **visual** skill presentation (v5 rendered
skill cards, 1000px, VLM-optimized fonts), SkillNet 3-stage flow, executed
by **Qwen3.5-9B** (`temperature=0`).

Skill selection is **NOT done by the model at runtime**. It is fixed to the
**top-5 skills per instance** from the qwen3.8-max annotation
`code/alfworld_skill_ranking_per_instance.json` (sorted by `relevance_score`).
Stages 2/3 (procedure text from skill-card **images** + procedure code) and
the agent loop are identical to `../text-skill` except for the skill
presentation.

## Layout

- `code/run_visual.py` — evaluation script (annotation-based retrieval + dynamic scheduling)
- `code/alfworld_skill_ranking_per_instance.json` — top-5 skill ranking from `annotation/outputs/`
- `code/base_config.yaml`, `code/alfworld_overall_procedure_examples.txt`,
  `code/alfworld_procedure_code_template.py` — SkillNet prompt examples and code template
- `images/skill_cards/` — the 37 v5 skill cards (not tracked; created by
  `scripts/render_skill_cards.sh`)
- `start_vllm.sh` / `stop_vllm.sh` — online vLLM deployment (4 servers, GPU i -> port 8000+i, TP=1)
- `launch.sh` — runs the eval with 128 dynamic workers pulling from one shared queue
- `results/alfworld/Qwen3.5-9B/vis_baseline/idx_*.json` — per-game results (written by `launch.sh`)

## Skill selection: matched by game_name

alfworld's `AlfredTWEnv` builds its game list via `os.walk()`, whose ordering
is not stable across dataset re-syncs; the annotation's `idx` order does NOT
match the env order, while the `task_name` sets match
exactly (134/134). `run_visual.py` therefore looks skills up **by game_name**
(idx is only a fallback with a warning), so every game gets exactly its
annotated top-5 skills regardless of env enumeration order.

## Run

```bash
bash ../../../../scripts/render_skill_cards.sh   # once: render the skill cards
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
