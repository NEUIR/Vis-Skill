#!/bin/bash
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
# Start vLLM servers for Qwen3.5-27B (text-only, no-skill baseline)
# 2 replicas, tensor-parallel-size=2 each.
# GPU(0,1) -> port 8000, GPU(2,3) -> port 8001.

MODEL="${MODELS_DIR:-$REPO_ROOT/models}/Qwen3.5-27B"
PYTHON="${PYTHON:-$(command -v python)}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

REPLICAS=(
    "0,1:8000"
    "2,3:8001"
)

echo "Starting 2 vLLM TP=2 instances for $MODEL (no-skill baseline) ..."

for spec in "${REPLICAS[@]}"; do
    GPUS="${spec%%:*}"
    PORT="${spec##*:}"
    LOG="$SCRIPT_DIR/vllm_${PORT}.log"
    echo "  GPUs $GPUS -> port $PORT (log: $LOG)"
    CUDA_VISIBLE_DEVICES=$GPUS $PYTHON -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --port "$PORT" \
        --tensor-parallel-size 2 \
        --max-model-len 24576 \
        --gpu-memory-utilization 0.90 \
        --no-enable-log-requests \
        > "$LOG" 2>&1 &
    echo "  PID: $!"
done

echo ""
echo "Waiting for servers to be ready..."
for spec in "${REPLICAS[@]}"; do
    PORT="${spec##*:}"
    while ! curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; do
        sleep 2
    done
    echo "  Port $PORT ready."
done
echo "All vLLM servers are up!"
