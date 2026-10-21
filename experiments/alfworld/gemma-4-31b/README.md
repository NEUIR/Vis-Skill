# ALFWorld x Gemma-4-31B — baseline-noskill / text-skill / vis-skill controlled experiments

Three controlled conditions on the Gemma-4-31B-it backbone (`models/Gemma-4-31B-it`):

| condition | folder | description |
|---|---|---|
| baseline-noskill | `baseline-noskill/` | no skill, the agent acts directly |
| text-skill | `text-skill/` | plain-text SKILL.md (skill.md only, no references/), full SkillNet three-stage pipeline; skill selection = top-5 of the qwen3.8-max annotation |
| vis-skill | `vis-skill/` | rendered skill card images (render_v5); same top-5 annotation as text-skill |

**Key constraint: vis-skill uses exactly the same skills per task as text-skill.**
Both read the top-5 of `code/alfworld_skill_ranking_per_instance.json`
(qwen3.8-max annotation). The selection is fixed in advance and does not depend
on any LLM retrieval, so the two conditions are aligned task by task.

## Running (online vLLM + dynamic scheduling)

All inference goes through **online vLLM OpenAI-compatible servers**. The 31B
bf16 weights take ~62.5 GB, which leaves almost no KV-cache headroom on a single
80 GB A100, so **each instance uses TP=2 (two GPUs), two instances on two ports**:

- GPU 0+1 → port 8000, GPU 2+3 → port 8001 (`--tensor-parallel-size 2`)
- `--gpu-memory-utilization 0.9`, `--max-model-len 24576`

The evaluator uses **dynamic scheduling**: all games go into one shared queue,
each worker process pulls the next game as soon as it finishes one, and the
vLLM port is assigned round-robin when a game is picked up, so every GPU stays
busy until the last game. Concurrency: baseline/text 64 workers, vis 48 workers.

```bash
cd experiments/alfworld/gemma-4-31b

# 0) environment: bash create_env.sh; self-check (no GPU needed):
bash verify_env.sh

# vis-skill needs the rendered cards (once, from the repository root):
#   bash scripts/render_skill_cards.sh

# Run all three conditions in order (start vLLM -> evaluate -> stop vLLM, per condition):
nohup bash run_all.sh > run_all.log 2>&1 &
#    run_all.sh refuses to start if other processes are on the GPUs, and between
#    conditions it only stops its own Gemma-4-31B vLLM servers.

# Or run each condition by hand (order: baseline -> text -> vis):
bash baseline-noskill/start_vllm.sh && bash baseline-noskill/launch.sh && bash baseline-noskill/stop_vllm.sh
bash text-skill/start_vllm.sh       && bash text-skill/launch.sh       && bash text-skill/stop_vllm.sh
bash vis-skill/start_vllm.sh        && bash vis-skill/launch.sh        && bash vis-skill/stop_vllm.sh
```

Outputs:
```
baseline-noskill/results/alfworld/Gemma-4-31B-it/noskill_baseline/summary_noskill_baseline.json
text-skill/results/alfworld/Gemma-4-31B-it/text_baseline/summary_text_baseline.json (+ skill_selections.json)
vis-skill/results/alfworld/Gemma-4-31B-it/vis_baseline/summary_vis_baseline.json
```

Evaluation is resume-safe: re-running launch.sh skips games that already have an
`idx_*.json` under `results/`.

## Gemma-4-31B specifics

1. **Thinking off**: the client scripts pass
   `chat_template_kwargs: {"enable_thinking": false}`.
2. **Visual token budget** (vis-skill): the vLLM servers run with
   `--mm-processor-kwargs '{"max_soft_tokens": 1120}'` (Gemma 4 supports
   70/140/280/560/1120; skill cards are dense rendered text, so the highest tier
   is used for legibility) and `--limit-mm-per-prompt '{"image": 6}'` (the
   stage-2 prompt holds at most 5 skill cards).
3. **TP=2**: two GPUs per instance, port pool `8000,8001`; the dynamic
   scheduling is unchanged.

## Model and data

- backbone: `models/Gemma-4-31B-it` (multimodal, bf16, ~62.5 GB); override with `MODELS_DIR`.
- dataset: `data/alfworld/` (split `eval_out_of_distribution`, 134 games); each
  launch.sh sets `ALFWORLD_DATA` to it unless already set.

## Layout

```
gemma-4-31b/
├── create_env.sh / verify_env.sh        # build the environment / self-check
├── run_all.sh                           # run all three conditions (GPU-occupancy guard + cleanup)
├── baseline-noskill/
│   ├── start_vllm.sh / stop_vllm.sh     # online vLLM (TP=2 x2, gpu-util 0.9, max-model-len 24576)
│   ├── launch.sh                        # dynamic-scheduling evaluator (64 workers)
│   └── code/                            # run_noskill.py + base_config.yaml
├── text-skill/
│   ├── start_vllm.sh / stop_vllm.sh
│   ├── launch.sh                        # dynamic-scheduling evaluator (64 workers)
│   ├── skills_alfworld/                 # the 37 skills (same as SkillNet)
│   └── code/                            # run_text.py + prompt examples / code template / annotation / base_config
└── vis-skill/
    ├── start_vllm.sh / stop_vllm.sh     # online multimodal vLLM (+ mm arguments)
    ├── launch.sh                        # dynamic-scheduling evaluator (48 workers)
    ├── skills_alfworld/                 # skill sources for the renderer
    └── code/                            # run_visual.py + render_v5.py + fonts + config
```

## Notes

- baseline/text-skill do not set sampling parameters in the request; vLLM's
  `--generation-config` is left at auto, so the model's generation_config.json
  defaults apply (temperature=1.0 / top_p=0.95 / top_k=64). vis-skill sets
  temperature=0 explicitly.
- All three conditions disable thinking (`enable_thinking=false`).
- Make sure the 4 GPUs are free before starting vLLM (run_all.sh checks this; when
  running by hand, check `nvidia-smi`). Ports default to 8000/8001 and can be
  changed via `PORTS/GPU_GROUPS` in the scripts (update `--ports` and the health
  check in launch.sh accordingly).
- To render the skill cards for this folder only:
  `python vis-skill/code/render_v5.py` (reads `vis-skill/skills_alfworld`, writes
  `vis-skill/images/skill_cards`).
