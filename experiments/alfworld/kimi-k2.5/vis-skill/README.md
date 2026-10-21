# ALFWorld / Kimi-K2.5 / vis-skill

Controlled ALFWorld experiment: **visual** skill presentation (v5 rendered
skill cards, 1000px, VLM-optimized fonts), SkillNet 3-stage flow, executed
by **kimi-k2.5** via the **DashScope API** (`temperature=0`,
same as the reference visual run).

Method/flow is identical to the Qwen3.5 ALFWorld experiments; only the
execution model and endpoint differ (DashScope OpenAI-compatible API), so no
local GPUs / vLLM servers are needed.

Skill selection is **NOT done by the model at runtime**. It is fixed to the
**top-5 skills per instance** from the qwen3.8-max annotation
`code/alfworld_skill_ranking_per_instance.json` (sorted by `relevance_score`),
so the controlled comparison against the other experiments stays valid.

## Layout

- `code/run_visual.py` — evaluation script (annotation-based retrieval +
  dynamic scheduling; multimodal stage-2 calls go to the DashScope
  OpenAI-compatible endpoint)
- `code/alfworld_skill_ranking_per_instance.json` — identical to the file of
  the same name in `../../qwen3.5-9b/vis-skill/code/`
- `code/base_config.yaml`, `code/alfworld_overall_procedure_examples.txt`,
  `code/alfworld_procedure_code_template.py` — identical to the files of the
  same name in `../../qwen3.5-9b/vis-skill/code/`
- `images/skill_cards/` — the 37 v5-rendered skill cards
- `launch.sh` — runs the eval with 20 dynamic workers pulling from one shared
  queue
- `results/alfworld/kimi-k2.5/vis_baseline/idx_*.json` — per-game results

## Skill selection: matched by game_name

`run_visual.py` looks skills up **by game_name** (idx is only a fallback with
a warning), so every game gets exactly its annotated top-5 skills regardless
of env enumeration order.

## API key

```bash
export DASHSCOPE_API_KEY=<your-key>
```

Model id: `kimi-k2.5` (must be vision-capable since stage 2 sends skill
cards as base64 images).

## Run

```bash
export DASHSCOPE_API_KEY=<your-key>
bash launch.sh       # evaluate 134 OOD games (resume-safe, skips existing idx_*.json)
```

Environment: `envs/environment_alfworld.yml`
(alfworld 0.4.2 + openai client). Data: `ALFWORLD_DATA=data/alfworld`.
