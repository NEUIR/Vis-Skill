# ALFWorld / Kimi-K2.5 / text-skill

Controlled ALFWorld experiment: **text** skill presentation (SKILL.md only),
SkillNet 3-stage flow, executed by **kimi-k2.5** via the **DashScope
API**.

Method/flow is identical to the Qwen3.5 ALFWorld experiments; only the
execution model and endpoint differ (DashScope OpenAI-compatible API), so no
local GPUs / vLLM servers are needed.

Skill selection is **NOT done by the model at runtime**. It is fixed to the
**top-5 skills per instance** from the qwen3.8-max annotation
`code/alfworld_skill_ranking_per_instance.json` (sorted by `relevance_score`),
so the controlled comparison against the other experiments stays valid.

## Layout

- `code/run_text.py` — evaluation script (annotation-based retrieval + dynamic
  scheduling; calls the DashScope OpenAI-compatible endpoint)
- `code/alfworld_skill_ranking_per_instance.json` — identical to the file of
  the same name in `../../qwen3.5-9b/text-skill/code/`
- `code/base_config.yaml`, `code/alfworld_overall_procedure_examples.txt`,
  `code/alfworld_procedure_code_template.py` — identical to the files of the
  same name in `../../qwen3.5-9b/text-skill/code/`
- `skills_alfworld/` — the 37 skills
- `launch.sh` — runs the eval with 20 dynamic workers pulling from one shared
  queue
- `results/alfworld/kimi-k2.5/text_baseline/idx_*.json` — per-game results

## API key

```bash
export DASHSCOPE_API_KEY=<your-key>
```

Model id: `kimi-k2.5`.

## Run

```bash
export DASHSCOPE_API_KEY=<your-key>
bash launch.sh       # evaluate 134 OOD games (resume-safe, skips existing idx_*.json)
```

Environment: `envs/environment_alfworld.yml`
(alfworld 0.4.2 + openai client). Data: `ALFWORLD_DATA=data/alfworld`.
