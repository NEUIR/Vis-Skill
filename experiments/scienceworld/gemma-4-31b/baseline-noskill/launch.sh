#!/bin/bash
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
# ============================================================================
# ScienceWorld baseline-noskill (Gemma-4-31B) — evaluation launcher.
#
# No skill is given to the model: system prompt + task description only,
# Thought/Action loop, SkillNet official ScienceWorld test split (211 eps).
# Reference flow: ../../../qwen3.5-9b/baseline-noskill.
#
# Dynamic scheduling: ALL episodes sit in ONE shared queue; 48 persistent
# workers each pull the NEXT episode the moment they finish their current
# one, and every LLM request round-robins across BOTH 2 online vLLM servers
# (ports 8000-8001) — all GPUs stay busy until the last task.
#
# The runner code lives in ../shared/code/ (single copy for all three
# experiments); datasets are read from ../shared/ (symlinks).
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-$(command -v python)}"
MODEL="${MODELS_DIR:-$REPO_ROOT/models}/Gemma-4-31B-it"
MODEL_NAME="Gemma-4-31B-it"
BASE_PORT=8000
NUM_PORTS=2
MAX_WORKERS=48       # shared-queue workers; high concurrency, each pulls next when free
MAX_STEPS=30
EPISODE_TIMEOUT=2400 # hard wall-clock limit per episode (seconds)
MAX_ATTEMPTS=2       # attempts per episode before an error result is recorded

RESULTS_DIR="$SCRIPT_DIR/results/scienceworld/$MODEL_NAME/noskill_baseline"

# Standalone convenience: check the shared online vLLM servers (started by
# ../start_vllm.sh, reused by all three 31B experiments).
for i in $(seq 0 $((NUM_PORTS - 1))); do
    PORT=$((BASE_PORT + i))
    if ! curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "[launch] vLLM on port $PORT not healthy — starting the shared servers..."
        bash "$SCRIPT_DIR/../start_vllm.sh"
        break
    fi
done

"$PYTHON" "$SCRIPT_DIR/../shared/code/run_scienceworld_baseline.py" \
    --model "$MODEL" \
    --output_path "$RESULTS_DIR" \
    --experiment_name "baseline-noskill-gemma-31B-scienceworld" \
    --max_workers "$MAX_WORKERS" \
    --max_steps "$MAX_STEPS" \
    --base_port "$BASE_PORT" \
    --num_ports "$NUM_PORTS" \
    --episode_timeout "$EPISODE_TIMEOUT" \
    --max_attempts "$MAX_ATTEMPTS" \
    2>&1 | tee "$SCRIPT_DIR/run.log"

echo "baseline-noskill (ScienceWorld, Gemma-4-31B) done. Log: $SCRIPT_DIR/run.log"
echo "Results: $RESULTS_DIR"
