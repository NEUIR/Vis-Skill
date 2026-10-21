# ALFWorld / Kimi-K2.5 / baseline-noskill

Controlled ALFWorld experiment: **no-skill baseline** (direct ReAct loop),
executed by **kimi-k2.5** via the **DashScope API**.

Method/flow is identical to the Qwen3.5 ALFWorld experiments; only the
execution model and endpoint differ (DashScope OpenAI-compatible API), so no
local GPUs / vLLM servers are needed.

The agent receives only the system prompt and task observation, then acts
step by step. No skill retrieval, no procedure generation — pure ReAct.

## Layout

- `code/run_noskill.py` — evaluation script (dynamic scheduling; calls the
  DashScope OpenAI-compatible endpoint)
- `code/base_config.yaml` — identical to `../../qwen3.5-9b/baseline-noskill/code/base_config.yaml`
- `launch.sh` — runs the eval with 20 dynamic workers pulling from one shared
  queue
- `results/alfworld/kimi-k2.5/noskill_baseline/idx_*.json` — per-game results

## API key

The script reads the DashScope API key from the environment variable
`DASHSCOPE_API_KEY` (base URL defaults to
`https://dashscope.aliyuncs.com/compatible-mode/v1`, override with
`DASHSCOPE_BASE_URL` if needed):

```bash
export DASHSCOPE_API_KEY=<your-key>          # add your own key
# optional: export DASHSCOPE_BASE_URL=...
```

Model id: `kimi-k2.5` (override with `--model <id>` if your DashScope
account exposes it under a different name; the result dir uses the last
`/`-segment of the model id).

## Run

```bash
export DASHSCOPE_API_KEY=<your-key>
bash launch.sh       # evaluate 134 OOD games (resume-safe, skips existing idx_*.json)
```

Scheduling: every worker, once it finishes a game, pulls the next game from a
single shared queue (true dynamic scheduling). Workers default to 20 — adjust
`--max_workers` in `launch.sh` if your API rate limit requires more or fewer.

Environment: `envs/environment_alfworld.yml`
(alfworld 0.4.2 + openai client). Data: `ALFWORLD_DATA=data/alfworld`.
