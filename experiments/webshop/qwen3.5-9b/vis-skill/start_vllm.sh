#!/bin/bash
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
# Start vLLM servers on 4 GPUs for Qwen3.5-9B (multimodal, high concurrency)
# vis-skill 9B variant: one vLLM instance per GPU (TP=1).
# Enables --limit-mm-per-prompt for visual skill card input (up to 5 images per request)

MODEL="${MODELS_DIR:-$REPO_ROOT/models}/Qwen3.5-9B"
PORTS=(8000 8001 8002 8003)
GPUS=(0 1 2 3)

PYTHON="${PYTHON:-$(command -v python)}"

echo "Starting 4 vLLM instances (multimodal) for $MODEL ..."

for i in "${!PORTS[@]}"; do
    PORT=${PORTS[$i]}
    GPU=${GPUS[$i]}
    LOG="/tmp/vllm_webshop_visskill_v1_9b_gpu${GPU}.log"
    echo "  GPU $GPU -> port $PORT (log: $LOG)"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --host 0.0.0.0 \
        --port "$PORT" \
        --trust-remote-code \
        --tensor-parallel-size 1 \
        --max-model-len 65536 \
        --gpu-memory-utilization 0.85 \
        --enforce-eager \
        --no-enable-log-requests \
        --limit-mm-per-prompt '{"image": 5}' \
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
echo "All vLLM servers are up!"
