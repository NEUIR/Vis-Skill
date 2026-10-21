#!/bin/bash
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
# =============================================================================
# Run the NO-SKILL baseline experiment with kimi-k2.5 (direct ReAct loop,
# no skill retrieval or procedure generation).
#
# Model calls go to the DashScope OpenAI-compatible API.
# The API key is read from $DASHSCOPE_API_KEY.
#
# Dynamic scheduling: 20 worker processes pull games from ONE shared queue.
# =============================================================================
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

export ALFWORLD_DATA="${ALFWORLD_DATA:-$REPO_ROOT/data/alfworld}"

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

${PYTHON:-python} code/run_noskill.py \
    --model kimi-k2.5 \
    --max_workers 20 \
    --max_steps 30 \
    > logs_run.log 2>&1

echo "Done. Check logs_run.log for details."
echo "Results: results/alfworld/kimi-k2.5/noskill_baseline/"
