#!/bin/bash
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
# ============================================================================
# ScienceWorld baseline-noskill (Qwen3.5-9B) — evaluation launcher.
#
# No skill is given to the model: system prompt + task description only,
# Thought/Action loop, SkillNet official ScienceWorld test split (211 eps).
#
# All episodes go through ONE shared worker queue (dynamic scheduling) and
# hit the single online vLLM endpoint on port 8000 (DP=4 across 4 GPUs).
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-$(command -v python)}"
MODEL="${MODELS_DIR:-$REPO_ROOT/models}/Qwen3.5-9B"
PORT=8000
MAX_WORKERS=32   # shared-queue workers; each pulls the next episode when free
MAX_STEPS=30

# Standalone convenience: start a local server if none is serving.
if ! curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
    bash "$SCRIPT_DIR/start_vllm.sh"
fi

"$PYTHON" code/run_scienceworld_baseline.py \
    --model "$MODEL" \
    --max_workers "$MAX_WORKERS" \
    --max_steps "$MAX_STEPS" \
    --port "$PORT" \
    > "$SCRIPT_DIR/run.log" 2>&1

echo "baseline-noskill (ScienceWorld) done. Log: $SCRIPT_DIR/run.log"
