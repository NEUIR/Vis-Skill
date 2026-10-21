# WebShop x Kimi-K2.5 — baseline / text-skill / vis-skill

Same three WebShop experiments (200 test sessions) and the same fixed top-5
skill selections as the Qwen3.5 / Gemma runs; `kimi-k2.5` is called through
the DashScope OpenAI-compatible API (no GPUs, no vLLM).

| Experiment | Directory | Runner |
|---|---|---|
| baseline-noskill | `baseline-noskill/` | `run_noskill.py` |
| text-skill | `text-skill/` | `run_textskill.py` (annotation top-5, SKILL.md text) |
| vis-skill | `vis-skill/` | `run_visskill.py` (`skill_selections.json`, skill-card images) |

Results: `<experiment>/results/kimi-k2.5/<exp_name>_skill_<True|False>/`.
Shared resources: `../web_agent_site/`, `../data`, `../search_engine`
(links created by `scripts/setup_data.sh`).

## Run

```bash
bash scripts/setup_data.sh                         # from the repo root
bash scripts/render_skill_cards.sh                 # from the repo root (vis-skill cards)
export DASHSCOPE_API_KEY=<your-key>
cd experiments/webshop/kimi-k2.5
nohup bash run_all.sh > run_all.out 2>&1 &         # baseline -> text-skill -> vis-skill
```

A single experiment: `bash <experiment>/launch.sh` (20 workers). Runs are resumable.
