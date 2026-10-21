#!/bin/bash
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
# Start vLLM servers on 4 GPUs for Qwen3.5-9B (text-only, no-skill baseline)
# Each GPU runs one replica with TP=1.

MODEL="${MODELS_DIR:-$REPO_ROOT/models}/Qwen3.5-9B"
PORTS=(8000 8001 8002 8003)
GPUS=(0 1 2 3)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-$(command -v python)}"

echo "Starting 4 vLLM instances for $MODEL (no-skill baseline) ..."

for i in "${!PORTS[@]}"; do
    PORT=${PORTS[$i]}
    GPU=${GPUS[$i]}
    LOG="$SCRIPT_DIR/vllm_${PORT}.log"
    echo "  GPU $GPU -> port $PORT (log: $LOG)"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --port "$PORT" \
        --tensor-parallel-size 1 \
        --max-model-len 16384 \
        --gpu-memory-utilization 0.90 \
        --no-enable-log-requests \
        > "$LOG" 2>&1 &
    echo "  PID: $!"
done

echo ""
echo "Waiting for servers to be ready..."
for PORT in "${PORTS[@]}"; do
    while ! curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; do
        sleep 2
    done
    echo "  Port $PORT ready."
done
echo "All vLLM servers are up!"
