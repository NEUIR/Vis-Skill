#!/bin/bash
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
# =============================================================================
# Run the BASELINE-NOSKILL experiment (Gemma-4-31B, no skill retrieval).
# Prerequisite: the online vLLM servers are up (bash start_vllm.sh).
#
# Dynamic scheduling: 64 worker processes pull games from ONE shared queue;
# each game is round-robin assigned to a vLLM port when a worker picks it up,
# so all 4 GPUs (2 TP=2 instances) stay busy until the last game finishes.
# =============================================================================
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ALFWorld data (HuggingFace download, see README "Data"); override with ALFWORLD_DATA
export ALFWORLD_DATA="${ALFWORLD_DATA:-$REPO_ROOT/data/alfworld}"

# quick health check of the vLLM servers
for PORT in 8000 8001; do
    if ! curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "[ERROR] vLLM server on port $PORT is not healthy. Run: bash start_vllm.sh"
        exit 1
    fi
done

${PYTHON:-python} code/run_noskill.py \
    --model ${MODELS_DIR:-$REPO_ROOT/models}/Gemma-4-31B-it \
    --max_workers 64 \
    --max_steps 30 \
    --ports "8000,8001" \
    > logs_run.log 2>&1

echo "Done. Check logs_run.log for details."
echo "Results: results/alfworld/Gemma-4-31B-it/noskill_baseline/"
