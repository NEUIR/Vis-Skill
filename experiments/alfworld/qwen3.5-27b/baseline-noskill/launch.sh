#!/bin/bash
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
# Run baseline-noskill-27B experiment
# - No skill retrieval, no procedure generation
# - Agent directly executes tasks with system prompt + observation only
# - Temperature=0, all stages on Qwen3.5-27B
# - Backed by 2 vLLM replicas (TP=2 each) started via start_vllm.sh, ports 8000/8001

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

export ALFWORLD_DATA="${ALFWORLD_DATA:-$REPO_ROOT/data/alfworld}"

${PYTHON:-python} code/run_noskill.py \
    --model ${MODELS_DIR:-$REPO_ROOT/models}/Qwen3.5-27B \
    --max_workers 32 \
    --max_steps 30 \
    --ports "8000,8001" \
    > run.log 2>&1

echo "Done. Check run.log for details."
