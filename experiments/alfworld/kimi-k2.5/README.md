# ALFWorld x Kimi-K2.5 — baseline / text-skill / vis-skill

Same three experiments and the same fixed top-5 skill selections as the
Qwen3.5 ALFWorld runs; the backbone `kimi-k2.5` is called through the DashScope OpenAI-compatible API, so no GPUs or vLLM servers are needed.

| Experiment | Directory | Results |
|---|---|---|
| baseline-noskill | `baseline-noskill/` | `results/alfworld/kimi-k2.5/noskill_baseline/` |
| text-skill | `text-skill/` | `results/alfworld/kimi-k2.5/text_baseline/` |
| vis-skill | `vis-skill/` | `results/alfworld/kimi-k2.5/vis_baseline/` |

## Run

```bash
bash scripts/render_skill_cards.sh                 # from the repo root (vis-skill cards)
export DASHSCOPE_API_KEY=<your-key>
cd experiments/alfworld/kimi-k2.5
nohup bash run_all.sh > run_all.log 2>&1 &         # baseline -> text-skill -> vis-skill
```

A single experiment: `cd <experiment> && bash launch.sh` (20 workers, one shared queue).
Runs are resumable. Details: `<experiment>/README.md`.
