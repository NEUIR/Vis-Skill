# ScienceWorld x Gemma-4-31B — baseline / text-skill / vis-skill

Three experiments (baseline-noskill / text-skill / vis-skill) with
Gemma-4-31B-it as the backbone.

| Part | Reference |
|---|---|
| text-skill / vis-skill (SkillNet 3-stage pipeline, fixed top-5 selection) | `experiments/scienceworld/qwen3.5-9b` |
| baseline-noskill (system prompt + task description, Thought/Action loop) | `experiments/scienceworld/qwen3.5-9b/baseline-noskill` |
| Gemma model / deployment settings | `experiments/alfworld/gemma-4-31b` |

- Backbone: `models/Gemma-4-31B-it`. The `-it` directory ships the official
  chat template, which vLLM loads directly.
- Data: the SkillNet ScienceWorld test split (211 episodes, 24 tasks,
  `simplificationStr="easy"`, max_steps=30). Requests use `temperature=0` and
  `chat_template_kwargs={"enable_thinking": false}`.
- Skill selection: the top-5 skills per episode from the Qwen3.8-Max
  annotation (`science_skill_annotations.json`). The selection does not depend
  on the backbone, so text-skill and vis-skill use exactly the same skills.

## Run

```bash
bash scripts/render_skill_cards.sh   # from the repo root: render the skill cards

cd experiments/scienceworld/gemma-4-31b
bash create_env.sh        # envs/vis-skill-science-gemma (once)
bash verify_env.sh        # environment checks (no GPU use)

nohup bash run_all.sh > run_all.nohup 2>&1 &
```

A single experiment can be run with e.g. `bash text-skill/launch.sh`
(it checks for, and if needed starts, the shared vLLM servers). Runs are
resumable: finished `idx_*.json` files are skipped.

## vLLM deployment and scheduling

All inference goes through the vLLM OpenAI-compatible server. One deployment
is started once and shared by all three experiments (the multimodal flags do
not affect text-only requests):

| Deployment | Ports |
|---|---|
| 2 instances, one per GPU pair (bf16 weights ~62.5 GB), TP=2, gpu-util 0.9, max-model-len 24576 | 8000-8001 |

- Multimodal flags: `--limit-mm-per-prompt '{"image": 8}'` (stage 2 sends at
  most 5 cards) and `--mm-processor-kwargs '{"max_soft_tokens": 1120}'` (the
  highest of Gemma 4's 70/140/280/560/1120 visual-token budgets, since the
  skill cards are dense text).
- No `--enforce-eager`, so vLLM uses CUDA graphs.

Scheduling works at two levels:
1. Episodes: all 211 episodes go into one shared queue; persistent workers
   take the next episode as soon as they finish one.
2. Requests: every LLM request is sent round-robin across the servers, with a
   random start offset per episode process.

Workers: 48 for baseline/text, 32 for vis (set at the top of each `launch.sh`).

Each episode runs in its own process with its own JVM. A hard wall-clock
timeout (2400 s for baseline/text, 1500 s for vis) kills the process tree;
the episode is retried up to 2 times and otherwise recorded as an error result.
The supervisor also restarts dead workers and checks for overdue in-flight
episodes.

## Shared data and code

Reusable inputs live once in `shared/`. `shared/setup_shared.sh` links them to
the copies in `experiments/scienceworld/qwen3.5-9b`:

- `test_indices.json` (211 episodes)
- `science_skill_annotations.json` (Qwen3.8-Max annotation)
- `scienceworld_overall_procedure_examples.txt`, `scienceworld_procedure_code_template.py`
- `skills_scienceworld/` (61 skills, SKILL.md only)
- `skill_cards_v6/` (61 skill cards, created by `scripts/render_skill_cards.sh`)
- `skill_selections.json` is generated, not linked, by
  `shared/code/prepare_skill_selections.py`.

There is one copy of the runner code, in `shared/code/`. Each `launch.sh`
passes `--model / --output_path / --experiment_name`:

```
shared/code/run_scienceworld_baseline.py   # baseline
shared/code/run_scienceworld_text.py       # text-skill (stage 1 = annotation top-5)
shared/code/run_scienceworld_vis.py        # vis-skill (stage 2 reads skill-card images)
shared/code/prepare_skill_selections.py    # build + cross-check the fixed selections
```

## Selection gates (`run_all.sh`)

1. **build**: read the annotation and take `ranked_skills[:5]` per episode.
   The run exits unless all 211 entries are present, task/variation match
   `test_indices.json`, and every selected skill exists in the local skill
   library (by SKILL.md frontmatter name).
2. baseline-noskill.
3. text-skill (the runner validates the annotation again before starting).
4. **cross-check**: the `relevant_skill_names` in every text-skill
   `idx_*.json` must match the annotation top-5 exactly, in order. The task
   query of each episode is written back into `skill_selections.json`.
5. vis-skill (reads `shared/skill_selections.json` and the skill cards).

## Summary metrics

Each runner writes `summary.json` with:
- `overall`: total / avg_score / avg_steps / task_done(_rate) / success_count / success_rate
- `by_task`: the same per task (success = reward >= 100)
- `by_category`: 8 task groups (State / Elec. / Class. / Growth / Chem. / Life / Stages / Meas.)

## Layout

```
experiments/scienceworld/gemma-4-31b/
├── run_all.sh                     # GPU-busy guard -> gates -> shared vLLM (2 x TP=2) -> three experiments
├── start_vllm.sh / stop_vllm.sh
├── create_env.sh / verify_env.sh  # envs/environment_scienceworld_gemma.yml (vllm 0.27.1, transformers 5.15.0, torch 2.13.0)
├── shared/
│   ├── setup_shared.sh
│   ├── test_indices.json, science_skill_annotations.json, *.txt, *.py, skills_scienceworld/
│   ├── skill_selections.json
│   └── code/                      # the four scripts above
├── baseline-noskill/launch.sh
├── text-skill/launch.sh
└── vis-skill/launch.sh
```

Results: `{exp}/results/scienceworld/Gemma-4-31B-it/{noskill_baseline|text_skill|vis_skill}/`.

## Gemma-specific notes

1. **Chat template / thinking**: the official Gemma 4 chat template from the
   `-it` model directory is used; every request sets
   `chat_template_kwargs: {"enable_thinking": false}`.
2. **Sampling**: every request sets `temperature=0` (SkillNet ScienceWorld protocol).
3. **TP=2**: the bf16 weights (~62.5 GB) leave almost no KV-cache headroom on
   one 80 GB GPU, so each instance uses two GPUs.
4. **Port safety**: `start_vllm.sh` / `run_all.sh` reuse a port only if it
   already serves this backbone; otherwise they exit with an error.
