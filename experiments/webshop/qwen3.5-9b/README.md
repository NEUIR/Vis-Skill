# WebShop x Qwen3.5-9B — baseline / text-skill / vis-skill

Three WebShop experiments (200 test sessions) with Qwen3.5-9B served by online
vLLM (4 instances, TP=1, ports 8000-8003). Both skill runs use the fixed
top-5 skills per session from the Qwen3.8-Max annotation
(`webshop_skill_annotations.json`); the model does not select skills.
The text-/vis-skill runs use the v1 prompts (ASIN-based product clicks).

| Experiment | Directory | Results |
|---|---|---|
| baseline-noskill | `baseline-noskill/` | `results/Qwen3.5-9B/qwen_webshop_baseline_skill_False/` |
| text-skill | `text-skill/` | `results/Qwen3.5-9B/qwen_webshop_textskill_v1_skill_True/` |
| vis-skill | `vis-skill/` | `results/Qwen3.5-9B/qwen9b_webshop_visskill_v1_skill_True/` |

Shared resources (one copy for all WebShop backbones): `../web_agent_site/`
and the `../data`, `../search_engine` links created by `scripts/setup_data.sh`.

## Run

```bash
bash scripts/setup_data.sh                         # from the repo root
bash scripts/render_skill_cards.sh                 # from the repo root (vis-skill cards)
cd experiments/webshop/qwen3.5-9b
nohup bash run_all.sh > run_all.out 2>&1 &         # baseline -> text-skill -> vis-skill
```

Each `launch.sh` checks the ports and starts its own vLLM servers if needed;
`run_all.sh` stops the servers between experiments. Workers: 12 (baseline-noskill), 8 (text-skill, vis-skill).

## Top-k series

`text-skill/topk/` and `vis-skill/topk/` rerun the skill experiments with
3/7/9/13/21 skills per session (top-5 is the main run):

```bash
bash text-skill/topk/run_all.sh                    # all levels, 9b
bash vis-skill/topk/run_all.sh
bash text-skill/topk/launch_level.sh 9             # one level
```

Environment: `envs/environment_webshop.yml`.
