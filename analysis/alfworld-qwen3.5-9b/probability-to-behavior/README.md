# Probability-to-Behavior — from distractor probability to executed actions

## Question

lexical-interference measures how keyword frequency k shifts the probability
of an action that is **not executable** in the current state. This experiment
closes the chain to behaviour: the competitor is taken from the real
`admissible_commands` — executable now, but not the correct next step (e.g.
`go to microwave 1` when the correct action is `take apple 1 from
countertop 1`). The model freely generates the next action, and we measure how
often it selects the competitor as k grows (Fig. 3c), alongside the candidate
probabilities under the same k-controlled documents.

## Layout

```
code/
  extract_behavior_states.py   # replay successful main-experiment trajectories; choose an admissible competitor per state
  construct_variants.py        # k = 0/1/2/4/8 documents (same machinery as lexical-interference)
  render_cards.py              # structural skill cards  -> variants/visual
  render_rasterized.py         # plain-text screenshot cards -> variants/rasterized
  run_probe.py                 # candidate probabilities + free generation per (state × k × rep)
  run_rollout.py               # optional env rollouts after the correct vs competitor branch
  analyze_results.py           # trends vs k, probability↔behaviour consistency, rollout cost
  smoke_test.py                # single-state end-to-end gate
  probe_lib.py / prompts.py / skill_lexicon.py / base_config.yaml
paper_figures/
  plot_fig3c_behavioral_realization.py   # Fig. 3c (reads results/analysis/per_trial.csv of both models)
launch.sh        # stage driver (extract, construct, render, render_raster, probe, rollout, analyze)
run_full.sh      # offline stages -> vLLM -> smoke gate -> probe -> analyze
run_rollout.sh   # rollout sweep (text k=0,1,2,4,8; structural k=0,4,8) -> analyze
start_vllm.sh / stop_vllm.sh
```

## Inputs

- successful trajectories of the ALFWorld main experiments
  (`experiments/alfworld/qwen3.5-9b/{text-skill,vis-skill}/results/...`);
- the skill corpus `experiments/alfworld/qwen3.5-9b/text-skill/skills_alfworld`;
- `$ALFWORLD_DATA` (default `data/alfworld`) for the env replay.

## Run

```bash
# Full pipeline
setsid nohup bash run_full.sh > /dev/null 2>&1 &
tail -f logs_full_run.log

# Optional rollouts (after the probe)
bash run_rollout.sh
```

## Outputs

- `states/`, `variants/{text,visual,rasterized}/` — built inputs (also used by
  `../../alfworld-gemma-4-31b/probability-to-behavior/`)
- `results/` — per-cell probe records; `results/analysis/` — merged analysis
- `rollout_results/` — rollout records
