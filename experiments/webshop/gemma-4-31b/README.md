# WebShop x Gemma-4-31B — baseline / text-skill / vis-skill

Three experiments on WebShop (200 test sessions) with Gemma-4-31B-it as the
backbone, served by online vLLM with dynamic scheduling.

| Experiment | Directory | Skill presentation | Reference flow |
|---|---|---|---|
| baseline-noskill | `baseline-noskill/` | no skill | `experiments/webshop/qwen3.5-9b/baseline-noskill` |
| text-skill | `text-skill/` | SKILL.md text only (no `references/`) | `experiments/webshop/qwen3.5-9b/text-skill` |
| vis-skill | `vis-skill/` | rendered skill-card images (`render_v7.py`) | `experiments/webshop/qwen3.5-9b/vis-skill` |

**Skill selection is not done by the model.** Both skill experiments use the
top-5 skills per session from the Qwen3.8-Max annotation
`annotation/outputs/webshop_skill_annotations.json`:
- text-skill: `run_gemma_textskill.py` reads the annotation file and takes the
  top-5 per session (the LLM retrieval stage is skipped);
- vis-skill: `prepare_skill_selections.sh` runs
  `code/generate_skill_selections.py` on the same file and writes
  `skill_selections.json`, so both experiments see exactly the same skills.

**Data** is shared, not copied per experiment. `scripts/setup_data.sh` links
`data/webshop/data` and `data/webshop/search_engine` into
`experiments/webshop/` (shared by all backbones, next to the shared
`web_agent_site/` simulator), and all three experiments read from there:

```
experiments/webshop/data/
├── items_shuffle.json, items_ins_v2.json, items_human_ins.json
├── webshop_overall_procedure_examples.txt   # stage-2 examples
└── webshop_procedure_code_template.py       # stage-3 code template
```

## Run

```bash
bash scripts/setup_data.sh            # from the repo root: link data/ and search_engine/
bash scripts/render_skill_cards.sh    # from the repo root: render the skill cards

cd experiments/webshop/gemma-4-31b
bash create_env.sh      # create envs/vis-skill-webshop-gemma (skipped if it exists)
bash verify_env.sh      # environment, data and model checks (no GPU use)

nohup bash run_all.sh > run_all.log 2>&1 &
```

`run_all.sh` runs **baseline -> text-skill -> skill-selection check -> vis-skill**.
For each experiment it starts two vLLM servers (TP=2 each), runs the
evaluation, then stops only its own servers (tracked by PID file). Runs are
resumable: sessions that already have `results/.../idx_*.json` are skipped.

The weights in `models/Gemma-4-31B-it` must be complete: `run_all.sh`,
`start_vllm.sh` and `verify_env.sh` check every shard listed in
`model.safetensors.index.json` and exit on a missing one.

## vLLM deployment

Gemma-4-31B (`Gemma4ForConditionalGeneration`, dedicated vision encoder) has
~62.5 GB of bf16 weights, which does not fit on one A100-80GB together with a
65K-token KV cache. Each instance therefore spans two GPUs, and two instances
serve the four GPUs:

| Instance | GPUs | Port |
|---|---|---|
| 0 | 0,1 | 8000 |
| 1 | 2,3 | 8001 |

- `--tensor-parallel-size 2`, `--gpu-memory-utilization 0.90`, `--max-model-len 65536`
- The tokenizer and chat template are read from the model directory. Every
  request sets `chat_template_kwargs: {"enable_thinking": false}`.
- vis-skill only: `--limit-mm-per-prompt '{"image": 6}'` and
  `--mm-processor-kwargs '{"max_soft_tokens": 1120}'` (the highest of Gemma 4's
  70/140/280/560/1120 visual-token budgets, since the skill cards are dense text).
- All LLM calls use `temperature=0`.

## Dynamic scheduling

All 200 sessions are submitted to one shared queue (a single
`ProcessPoolExecutor`). A worker takes the next session as soon as it finishes
one, and requests are spread round-robin over the two ports, so all GPUs stay
busy until the last session. If the worker pool crashes, it is restarted and
unfinished sessions are rerun.

Concurrency is **12 workers**, the memory-safe limit for WebShop: each worker
loads its own copy of the product database (~16 GB RSS). Environment
initialisation is additionally limited by `Semaphore(3)`.

## Results and logs

```
baseline-noskill/results/Gemma-4-31B-it/gemma_webshop_baseline_skill_False/   idx_*.json + summary.json
text-skill/results/Gemma-4-31B-it/gemma_webshop_textskill_skill_True/         idx_*.json + summary.json
vis-skill/results/Gemma-4-31B-it/gemma_webshop_visskill_skill_True/           idx_*.json + summary.json
vis-skill/skill_selections.json                                           built from the annotation
```

`summary.json` reports score (mean reward x 100), success rate and average
steps. text-skill records the skills used per session in
`relevant_skill_names`. Evaluation logs: `<experiment>/logs_run.log`; vLLM logs:
`<experiment>/vllm_<port>.log`.

## Layout

```
experiments/webshop/
├── web_agent_site/                     # WebShop simulator (shared by all backbones)
├── data/, search_engine/               # links created by scripts/setup_data.sh
└── gemma-4-31b/
    ├── README.md / create_env.sh / verify_env.sh / run_all.sh
    ├── baseline-noskill/
    │   ├── start_vllm.sh / stop_vllm.sh / launch.sh
    │   ├── run_gemma_webshop.py
    │   └── prompts/ spacy_patch.py test_indices.json
    ├── text-skill/
    │   ├── start_vllm.sh / stop_vllm.sh / launch.sh
    │   ├── run_gemma_textskill.py      # annotation top-5 -> stages 2/3 (SKILL.md only)
    │   ├── prompt_generator.py         # stage-2/3 prompts
    │   ├── skills/                     # 23 SkillNet skills
    │   └── prompts/ spacy_patch.py test_indices.json
    └── vis-skill/
        ├── start_vllm.sh (multimodal flags) / stop_vllm.sh / launch.sh
        ├── prepare_skill_selections.sh
        ├── code/generate_skill_selections.py code/run_visual.py code/render_v7.py
        ├── skill_selections.json
        ├── images/skill_cards_v7/      # created by scripts/render_skill_cards.sh
        ├── fonts/ skills/
        └── prompts/ spacy_patch.py test_indices.json
```

## Notes

1. **Java**: the Lucene search (pyserini) needs a JVM. `JAVA_HOME` points to
   the `jdk4py` runtime in the environment (set in the Python scripts and in
   `launch.sh`).
2. **spaCy**: the reward function needs `en_core_web_sm`; `spacy_patch.py`
   replaces it with lightweight rules (same as the Qwen reference).
3. `run_all.sh` refuses to start if any GPU has more than 5 GB in use.
   `start_vllm.sh` exits if a port is held by another experiment and only
   reclaims ports held by a leftover `Gemma-4-31B-it` server.
4. Environment: `envs/vis-skill-webshop-gemma` (python 3.10, vllm 0.26.0,
   transformers 5.14.1, torch 2.11.0); see `create_env.sh` (`envs/environment_webshop_gemma.yml`).
