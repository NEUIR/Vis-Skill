# ScienceWorld × Qwen3.5-27B — CONTROLLED text-skill / vis-skill (qwen3.8-max annotation top-5)

Controlled comparison with **exactly the same pipeline as `../qwen3.5-9b`**
(SkillNet v1 pipeline + stage-1 fixed to the per-episode **top-5** skills of the
offline qwen3.8-max annotation). The **only change is the backbone**:
`models/Qwen3.5-27B` instead of `models/Qwen3.5-9B`.
text-skill and vis-skill use **exactly the same** fixed selection
(`skill_selections.json` is the same file as in the 9B experiment), so the text
vs vis comparison reflects only how the skill is presented.

| condition | folder | skill form |
|---|---|---|
| text-skill | `text-skill/` | text SKILL.md (no reference files) |
| vis-skill  | `vis-skill/`  | rendered skill card images (`images/skill_cards_v6/`, named by frontmatter name, 61 cards) |
| baseline   | `baseline-noskill/` | no skill |

Backbone: `models/Qwen3.5-27B`. Data: the official SkillNet ScienceWorld test
split (211 episodes / 24 tasks, `simplificationStr="easy"`, max_steps=30,
temperature=0).

## Running

```bash
# once, from the repository root: render the skill cards
bash scripts/render_skill_cards.sh

cd experiments/scienceworld/qwen3.5-27b
nohup bash run_all.sh > run_all.nohup 2>&1 &
tail -f run_all.log
```

Order: **baseline-noskill** (its own single-endpoint vLLM server, stopped
afterwards) → start the shared online vLLM for the skill runs → **build and validate
skill_selections.json from the annotation (hard gate)** → text-skill →
**cross-check that the text run really used the annotated top-5 for every
episode (hard gate)** → vis-skill → print all three summaries.
Everything is resume-safe (finished `idx_*.json` files are skipped).
The baseline alone: `cd baseline-noskill && bash launch.sh && bash stop_vllm.sh`.

## vLLM deployment (online, CUDA graph, dynamic scheduling)

- **2 online vLLM servers** (each `--tensor-parallel-size 2`, together using all
  4 A100-80GB; ports 8000-8001, OpenAI-compatible `/v1` endpoint),
  `gpu-memory-utilization 0.90`, `max-model-len 32768`; vis needs
  `--limit-mm-per-prompt '{"image": 8}'`. 27B does not fit on one GPU, hence
  TP=2; `--enforce-eager` is not set, so **CUDA graphs stay on** (vLLM default).
- Client: each experiment submits all 211 episodes to **one shared process-pool
  queue**; a worker pulls the next episode as soon as it finishes one (text and
  vis both 32 workers), and LLM requests are round-robined over all servers
  (each episode process starts the rotation at a random offset so they do not hit
  the same port in phase) → two-level (episode + request) dynamic scheduling.
- Hang protection: hard per-episode wall-clock timeout (text 2400 s / vis
  1200 s, twice the 9B vis limit for the slower 27B decoding) + process-tree
  SIGKILL + retries (at most 2); timeouts/failures are recorded as error
  results, so the run always terminates.
- Port guard: if ports 8000-8001 are already served by **another model** (e.g.
  the 9B experiment), the scripts exit with an error instead of silently
  reusing them.

## Building and checking the controlled selection

- `vis-skill/code/prepare_skill_selections.py` (identical to 9B)
  - **build mode**: read the annotation → take `ranked_skills[:5]` per episode;
    hard checks that all 211 entries exist, task/variation match
    `test_indices.json`, and every selected skill exists in the local skill
    library (by SKILL.md frontmatter name); any failure exits.
  - **cross-check mode** (`--text-results`): checks that `relevant_skill_names`
    of every text-skill `idx_*.json` **equals** the annotated top-5 (including
    order), and writes each episode's task query back into
    `skill_selections.json`.
- `run_scienceworld_text.py` validates the annotation again before starting
  (completeness + skill names in the library) and exits on failure.

## Summary metrics (built in)

`summary.json` contains (same definitions as the 9B experiment):

- `overall`: total / avg_score / avg_steps / task_done(_rate) / success_count / success_rate
- `by_task`: the same per task (success = reward ≥ 100)
- `by_category`: 8 task groups (State / Elec. / Class. / Growth / Chem. /
  Life / Stages / Meas., mapped by longest task-name prefix) with avg_score /
  success_count / success_rate / avg_steps

## Layout

```
qwen3.5-27b/
  run_all.sh start_vllm.sh stop_vllm.sh   # orchestration + shared vLLM
  baseline-noskill/
    launch.sh start_vllm.sh stop_vllm.sh
    code/  run_scienceworld_baseline.py
  text-skill/
    launch.sh start_vllm.sh stop_vllm.sh
    code/  run_scienceworld_text.py            # stage-1 = annotated top-5 (27B)
           science_skill_annotations.json      # annotation
           test_indices.json + examples/templates
    skills_scienceworld/                       # 61 skills, SKILL.md only
    skill_selections.json                      # exported after the run (diagnostics)
  vis-skill/
    launch.sh render.sh start_vllm.sh stop_vllm.sh
    skill_selections.json (+ _report.txt)      # controlled selection (same as 9B)
    code/  run_scienceworld_vis.py             # 27B
           prepare_skill_selections.py         # annotation-driven (build + cross-check)
           render_skill_cards.py + fonts/      # card renderer (v6)
           science_skill_annotations.json test_indices.json + examples/templates
    skills_scienceworld/
    images/skill_cards_v6/*.png                # created by the renderer
```

Results are written to `{text,vis}-skill/results/scienceworld/Qwen3.5-27B/{text,vis}_skill/`
and `baseline-noskill/results/scienceworld/Qwen3.5-27B/noskill_baseline/`.
