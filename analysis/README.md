# Analysis experiments

All probes run on ALFWorld. Inputs are built from the successful trajectories
of the Qwen3.5-9B main experiments, so those must be run first. Every stage
writes its outputs inside its own directory (`states/`, `variants/`,
`results/`); nothing here ships with data or results.

## Run order

| step | directory | needs | paper |
|---|---|---|---|
| 0 | `../experiments/alfworld/qwen3.5-9b/{text-skill,vis-skill}` | ALFWorld data, Qwen3.5-9B | Table 1 |
| 1 | `alfworld-qwen3.5-9b/lexical-interference` | step 0 trajectories + `text-skill/skills_alfworld` | Fig. 1a, 1b |
| 2 | `alfworld-qwen3.5-9b/causal-keyword-ablation` | step 1 `states/`, `variants/` | Fig. 1c |
| 3 | `alfworld-qwen3.5-9b/rasterized-interference` | step 1 `states/`, `variants/`, `results/` | Fig. 3a |
| 4 | `alfworld-qwen3.5-9b/probability-to-behavior` | step 0 trajectories | Fig. 3c |
| 5 | `alfworld-gemma-4-31b/probability-to-behavior` | step 4 `states/`, `variants/` | Fig. 3c |
| 6 | `alfworld-qwen3.5-9b/mechanism/symbol-encoding` | steps 1–2 | Fig. 3b |
| 7 | `alfworld-qwen3.5-9b/mechanism/layerwise-exposure-slope` | steps 1–2 | Fig. 4 |
| 8 | `alfworld-qwen3.5-9b/mechanism/component-patching` | steps 1–2 | Fig. 5 |
| 9 | `alfworld-gemma-4-31b/mechanism/{symbol-encoding,component-patching}` | steps 1–2 | Fig. 3b, 5 |

Steps 1–5 each provide `run_full.sh` (offline stages → vLLM → smoke gate →
probe → analyze) and `launch.sh` (single stages via `STAGES=...`). The
mechanism experiments load the model through HuggingFace and use
`launch.sh` only.

## Paper figures

Each experiment's `paper_figures/plot_fig*.py` reads the results written by
that experiment (default paths point into this repository). After the runs
above have finished, all figures are regenerated with:

```bash
bash ../scripts/make_figures.sh
```
