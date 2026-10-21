#!/bin/bash
# ============================================================================
# ScienceWorld baseline-noskill (kimi-k2.5) — evaluation launcher.
#
# No skill is given to the model: system prompt + task description only,
# Thought/Action loop, SkillNet official ScienceWorld test split (211 eps).
# Flow reference: scienceworld/gemma-4-31b (shared runner); endpoint reference:
# alfworld/kimi-k2.5 — DashScope OpenAI-compatible API, NO local vLLM.
#
# The API key is read from $DASHSCOPE_API_KEY (export it before running;
# the base URL can be overridden via $DASHSCOPE_BASE_URL).
#
# Dynamic scheduling: ALL episodes sit in ONE shared queue; 20 persistent
# workers each pull the NEXT episode the moment they finish their current
# one against the shared API endpoint.
#
# The runner code lives in ../shared/code/ (single copy for all three
# experiments); datasets are read from ../shared/ (symlinks).
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-$(command -v python)}"
MODEL="kimi-k2.5"
MODEL_NAME="kimi-k2.5"
MAX_WORKERS=20       # shared-queue workers (API concurrency; adjustable)
MAX_STEPS=30
TEMPERATURE=0.0      # SkillNet protocol: temperature=0 on the request side
EPISODE_TIMEOUT=2400 # hard wall-clock limit per episode (seconds)
MAX_ATTEMPTS=2       # attempts per episode before an error result is recorded

RESULTS_DIR="$SCRIPT_DIR/results/scienceworld/$MODEL_NAME/noskill_baseline"

# --- DashScope API key check ---
if [ -z "${DASHSCOPE_API_KEY:-}" ]; then
    echo "[ERROR] DASHSCOPE_API_KEY is not set."
    echo "        Export your own DashScope API key first, e.g.:"
    echo "        export DASHSCOPE_API_KEY=<your-key>"
    exit 1
fi

# quick API connectivity check
BASE_URL="${DASHSCOPE_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
    -H "Authorization: Bearer $DASHSCOPE_API_KEY" \
    "$BASE_URL/models")
if [ "$HTTP_CODE" != "200" ]; then
    echo "[WARN] DashScope API check returned HTTP $HTTP_CODE ($BASE_URL/models)."
    echo "       Continuing anyway; if the key/network is bad the run will fail."
fi

"$PYTHON" "$SCRIPT_DIR/../shared/code/run_scienceworld_baseline.py" \
    --model "$MODEL" \
    --output_path "$RESULTS_DIR" \
    --experiment_name "baseline-noskill-kimi-k2.5-scienceworld" \
    --max_workers "$MAX_WORKERS" \
    --max_steps "$MAX_STEPS" \
    --temperature "$TEMPERATURE" \
    --episode_timeout "$EPISODE_TIMEOUT" \
    --max_attempts "$MAX_ATTEMPTS" \
    2>&1 | tee "$SCRIPT_DIR/run.log"

echo "baseline-noskill (ScienceWorld, $MODEL, DashScope API) done. Log: $SCRIPT_DIR/run.log"
echo "Results: $RESULTS_DIR"
