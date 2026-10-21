#!/bin/bash
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
set -u

MODEL="${MODELS_DIR:-$REPO_ROOT/models}/Gemma-4-31B-it"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-$(command -v python)}"

PORTS=(8000 8001)
GPU_GROUPS=("0,1" "2,3")
GPU_UTIL=0.9
MAX_MODEL_LEN=24576
TP=2

if [ ! -x "$PYTHON" ]; then
    echo "[ERROR] $PYTHON not found"
    exit 1
fi

PID_FILE="$SCRIPT_DIR/vllm_pids.txt"
: > "$PID_FILE"

echo "Starting ${#PORTS[@]} vLLM instances (TP=$TP) for $MODEL (vis-skill) ..."
for i in "${!PORTS[@]}"; do
    PORT=${PORTS[$i]}
    GPUS=${GPU_GROUPS[$i]}
    LOG="$SCRIPT_DIR/vllm_${PORT}.log"
    echo "  GPU $GPUS -> port $PORT (log: $LOG)"
    CUDA_VISIBLE_DEVICES=$GPUS "$PYTHON" -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --port "$PORT" \
        --tensor-parallel-size "$TP" \
        --max-model-len "$MAX_MODEL_LEN" \
        --gpu-memory-utilization "$GPU_UTIL" \
        --limit-mm-per-prompt '{"image": 6}' \
        --mm-processor-kwargs '{"max_soft_tokens": 1120}' \
        --no-enable-log-requests \
        > "$LOG" 2>&1 &
    echo $! >> "$PID_FILE"
    echo "  PID: $!"
done

echo ""
echo "Waiting for servers to be ready..."
for PORT in "${PORTS[@]}"; do
    WAITED=0
    while ! curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; do
        sleep 5
        WAITED=$((WAITED + 5))
        if [ "$WAITED" -ge 1800 ]; then
            echo "[ERROR] port $PORT not healthy after 1800s — check $SCRIPT_DIR/vllm_${PORT}.log"
            exit 1
        fi
    done
    echo "  Port $PORT ready."
done
echo "All vLLM servers are up!"
