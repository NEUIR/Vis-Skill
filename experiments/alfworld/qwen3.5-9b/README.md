# ALFWorld x Qwen3.5-9B — baseline / text-skill / vis-skill

Three ALFWorld experiments (134 OOD games) with Qwen3.5-9B as the backbone.
Both skill runs use the fixed top-5 skills per game from the Qwen3.8-Max
annotation (`code/alfworld_skill_ranking_per_instance.json`); the model does
not select skills.

| Experiment | Directory | Skill presentation | Results |
|---|---|---|---|
| baseline-noskill | `baseline-noskill/` | no skill | `results/alfworld/Qwen3.5-9B/noskill_baseline/` |
| text-skill | `text-skill/` | SKILL.md text | `results/alfworld/Qwen3.5-9B/text_baseline/` |
| vis-skill | `vis-skill/` | rendered skill cards (`render_v5.py`) | `results/alfworld/Qwen3.5-9B/vis_baseline/` |

## Run

```bash
bash scripts/render_skill_cards.sh                 # from the repo root (vis-skill cards)
cd experiments/alfworld/qwen3.5-9b
nohup bash run_all.sh > run_all.log 2>&1 &         # baseline -> text-skill -> vis-skill
```

Each experiment starts its own vLLM servers (4 replicas, TP=1, GPU i -> port 8000+i), runs, and stops them.

A single experiment: `cd <experiment> && bash start_vllm.sh && bash launch.sh && bash stop_vllm.sh`.
Workers: 32 (baseline-noskill), 128 (text-skill, vis-skill), one shared queue, round-robin over ports. All runs are resumable (finished `idx_*.json` are skipped).
Details per experiment: `text-skill/README.md`, `vis-skill/README.md`.
Environment: `envs/environment_alfworld.yml`; data: `ALFWORLD_DATA=data/alfworld`.
