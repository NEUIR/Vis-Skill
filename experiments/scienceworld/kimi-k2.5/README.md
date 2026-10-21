# ScienceWorld x Kimi-K2.5 — baseline / text-skill / vis-skill (DashScope API)

Three experiments (baseline-noskill / text-skill / vis-skill) with Kimi-K2.5
as the backbone. The experiment flow is the same as
`experiments/scienceworld/gemma-4-31b`; the API usage is the same as
`experiments/alfworld/kimi-k2.5`.

- Model: `kimi-k2.5` (configurable via `--model`).
- Endpoint: DashScope OpenAI-compatible API
  - `base_url`: `$DASHSCOPE_BASE_URL` (default `https://dashscope.aliyuncs.com/compatible-mode/v1`)
  - `api_key`: `$DASHSCOPE_API_KEY`
  - No local vLLM and no GPU use.
- Data: the SkillNet ScienceWorld test split (211 episodes, 24 tasks,
  `simplificationStr="easy"`, max_steps=30, `temperature=0`).
- Skill selection: the top-5 skills per episode from the Qwen3.8-Max
  annotation (`science_skill_annotations.json`). text-skill and vis-skill use
  exactly the same skills.
- Concurrency: 20 persistent workers pull episodes from one shared queue
  (`MAX_WORKERS=20` at the top of each `launch.sh`).

## Run

```bash
bash scripts/render_skill_cards.sh        # from the repo root: render the skill cards

cd experiments/scienceworld/kimi-k2.5
export DASHSCOPE_API_KEY=<your-key>          # required
nohup bash run_all.sh > run_all.nohup 2>&1 &
tail -f run_all.log
```

A single experiment can be run with e.g. `bash text-skill/launch.sh`. Runs are
resumable: finished `idx_*.json` files are skipped.

Environment: `envs/environment_scienceworld.yml` (scienceworld, openai,
retry, pyyaml, psutil, tqdm). ScienceWorld needs `java` on `PATH`;
`run_all.sh` checks for it.

## Scheduling and timeouts

1. **Dynamic scheduling**: all 211 episodes go into one shared queue; each of
   the 20 workers takes the next episode as soon as it finishes one.
2. **Timeouts**: each episode runs in its own process and JVM (own session via
   `os.setsid()`). A hard wall-clock timeout (2400 s for baseline/text, 1800 s
   for vis) kills the process tree; the episode is retried up to
   `max_attempts=2` times and otherwise recorded as an error result. The
   supervisor also restarts dead workers and checks for overdue in-flight
   episodes.
3. **API retries**: every LLM request uses `@retry(tries=5, delay=5, backoff=2)`.

## Shared data and code

Reusable inputs live once in `shared/`. `shared/setup_shared.sh` links them to
the copies in `experiments/scienceworld/qwen3.5-9b`:

- `test_indices.json` (211 episodes)
- `science_skill_annotations.json` (Qwen3.8-Max annotation)
- `scienceworld_overall_procedure_examples.txt`, `scienceworld_procedure_code_template.py`
- `skills_scienceworld/` (61 skills, SKILL.md only)
- `skill_cards_v6/` (61 skill cards, created by `scripts/render_skill_cards.sh`;
  vis-skill sends them as base64 data URLs)
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
experiments/scienceworld/kimi-k2.5/
├── run_all.sh                     # gates -> baseline -> text -> cross-check -> vis
├── shared/
│   ├── setup_shared.sh
│   ├── test_indices.json, science_skill_annotations.json, *.txt, *.py, skills_scienceworld/
│   ├── skill_selections.json
│   └── code/                      # the four scripts above
├── baseline-noskill/launch.sh
├── text-skill/launch.sh
└── vis-skill/launch.sh
```

Results: `{exp}/results/scienceworld/kimi-k2.5/{noskill_baseline|text_skill|vis_skill}/`.

## API-specific notes

1. **Sampling**: every request sets `temperature=0` (SkillNet ScienceWorld
   protocol); a negative `--temperature` omits the parameter.
2. **Multimodal**: vis-skill stage 2 sends the 5 skill cards as
   `data:image/png;base64,...` URLs in the messages (same as
   `experiments/alfworld/kimi-k2.5/vis-skill`), so the model must be vision-capable.
3. **Rate limits**: on HTTP 429, lower `MAX_WORKERS` or rely on the
   request-level `@retry` backoff.
