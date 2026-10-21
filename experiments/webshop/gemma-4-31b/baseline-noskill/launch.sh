#!/bin/bash
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
# =============================================================================
# Run the BASELINE-NOSKILL experiment (Gemma-4-31B, no skill retrieval).
# Prerequisite: the online vLLM servers are up (bash start_vllm.sh).
#
# Deployment: 2 ONLINE vLLM instances x TP=2 (GPUs 0,1 -> port 8000,
# GPUs 2,3 -> port 8001).
#
# Dynamic scheduling: 12 worker processes pull games from ONE shared queue;
# whenever a worker finishes a game it immediately pulls the next one, and
# games are round-robin assigned to the 2 vLLM ports at worker init, so all
# 4 GPUs stay busy until the last game finishes.
# (12 workers = RAM-safe max: each worker holds a ~16GB product-DB copy;
#  the container cgroup limit is 256GB.)
# =============================================================================
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-$(command -v python)}"
if [ ! -x "$PYTHON" ]; then
    echo "[ERROR] $PYTHON not found — run: bash ../create_env.sh"
    exit 1
fi

export API_KEY=EMPTY
export JAVA_HOME="${JAVA_HOME:-$("${PYTHON:-python}" -c 'import jdk4py; print(jdk4py.JAVA_HOME)')}"

# quick health check of the vLLM servers
for PORT in 8000 8001; do
    if ! curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "[ERROR] vLLM server on port $PORT is not healthy. Run: bash start_vllm.sh"
        exit 1
    fi
done

"$PYTHON" "$SCRIPT_DIR/run_gemma_webshop.py" \
    --model ${MODELS_DIR:-$REPO_ROOT/models}/Gemma-4-31B-it \
    --max_workers 12 \
    --max_steps 30 \
    --exp_name gemma_webshop_baseline \
    --ports "8000,8001" \
    2>&1 | tee "$SCRIPT_DIR/logs_run.log"

echo "Done. Results: results/Gemma-4-31B-it/gemma_webshop_baseline_skill_False/"
