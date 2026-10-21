#!/bin/bash
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
# =============================================================================
# Run the TEXT-SKILL controlled experiment with kimi-k2.5 (SkillNet flow,
# SKILL.md only, skills fixed to the top-5 of the qwen3.8-max annotation).
#
# Model calls go to the DashScope OpenAI-compatible API.
# The API key is read from $DASHSCOPE_API_KEY.
#
# Dynamic scheduling: N worker processes pull games from ONE shared queue.
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

${PYTHON:-python} code/run_text.py \
    --model kimi-k2.5 \
    --use_skill \
    --max_workers 20 \
    --max_steps 30 \
    > logs_run.log 2>&1

echo "Done. Check logs_run.log for details."
echo "Results: results/alfworld/kimi-k2.5/text_baseline/"
