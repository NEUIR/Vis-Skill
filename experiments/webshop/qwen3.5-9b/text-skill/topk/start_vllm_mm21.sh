#!/bin/bash
# Start vLLM servers for the v1 top-k series (4 GPUs, one server each).
# Multimodal config with --limit-mm-per-prompt image=21 so that the largest
# level (top-21 skill cards) fits in one request; text-only requests are
# unaffected, so one server config serves both text-topk and vis-topk.
#
# Usage: bash start_vllm_mm21.sh <MODEL_PATH> [MAX_MODEL_LEN]
set -e

MODEL="${1:?usage: start_vllm_mm21.sh MODEL_PATH [MAX_MODEL_LEN]}"
MAX_MODEL_LEN="${2:-65536}"
PORTS=(8000 8001 8002 8003)
GPUS=(0 1 2 3)
MARKER="/tmp/vllm_webshop_v1_topk_mm21_marker.txt"

PYTHON="${PYTHON:-$(command -v python)}"
TAG=$(basename "$MODEL" | tr 'A-Z' 'a-z')

echo "Starting 4 vLLM instances (multimodal, image=21) for $MODEL (max-model-len $MAX_MODEL_LEN) ..."

for i in "${!PORTS[@]}"; do
    PORT=${PORTS[$i]}
    GPU=${GPUS[$i]}
    LOG="/tmp/vllm_webshop_v1_topk_${TAG}_gpu${GPU}.log"
    echo "  GPU $GPU -> port $PORT (log: $LOG)"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --host 0.0.0.0 \
        --port "$PORT" \
        --trust-remote-code \
        --tensor-parallel-size 1 \
        --max-model-len "$MAX_MODEL_LEN" \
        --gpu-memory-utilization 0.85 \
        --enforce-eager \
        --no-enable-log-requests \
        --limit-mm-per-prompt '{"image": 21}' \
        > "$LOG" 2>&1 &
    echo "  PID: $!"
done

echo ""
echo "Waiting for servers to be ready..."
for PORT in "${PORTS[@]}"; do
    for i in $(seq 1 120); do
        if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
            echo "  Port $PORT ready."
            break
        fi
        sleep 5
    done
done
echo "${MODEL}:${MAX_MODEL_LEN}" > "$MARKER"
echo "All vLLM servers are up! (marker: $MARKER)"
