#!/bin/bash
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
# =============================================================================
# Run the VIS-SKILL controlled experiment (Qwen3.5-27B, SkillNet flow,
# v5 rendered skill cards, skills fixed to the top-5 of the qwen3.8-max
# annotation). Prerequisite: the online vLLM servers are up (bash start_vllm.sh).
#
# Dynamic scheduling: 128 worker processes pull games from ONE shared queue;
# each game is round-robin assigned to a vLLM port when a worker picks it up,
# so both TP=2 replicas stay busy until the last game finishes.
# =============================================================================
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

export ALFWORLD_DATA="${ALFWORLD_DATA:-$REPO_ROOT/data/alfworld}"

# quick health check of the vLLM servers
for PORT in 8000 8001; do
    if ! curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "[ERROR] vLLM server on port $PORT is not healthy. Run: bash start_vllm.sh"
        exit 1
    fi
done

${PYTHON:-python} code/run_visual.py \
    --model ${MODELS_DIR:-$REPO_ROOT/models}/Qwen3.5-27B \
    --use_skill \
    --max_workers 128 \
    --max_steps 30 \
    --ports "8000,8001" \
    > logs_run.log 2>&1

echo "Done. Check logs_run.log for details."
echo "Results: results/alfworld/Qwen3.5-27B/vis_baseline/"
