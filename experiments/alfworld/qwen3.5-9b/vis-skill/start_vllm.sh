#!/bin/bash
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
# =============================================================================
# Start ONLINE vLLM servers for Qwen3.5-9B (multimodal) — one instance per
# GPU (TP=1).
#
#   - 4 servers: GPU i -> port 8000+i
#   - serving flags identical to ../text-skill/start_vllm.sh
#     (max-model-len 24576, gpu-util 0.90, limit-mm-per-prompt image:6)
#   - clients (code/run_visual.py) disable thinking via
#     chat_template_kwargs.enable_thinking=false
#
# If servers from a previous experiment occupy these ports/GPUs, stop them
# first (bash stop_vllm.sh there), they are independent processes.
# =============================================================================
set -u

MODEL="${MODELS_DIR:-$REPO_ROOT/models}/Qwen3.5-9B"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-$(command -v python)}"

PORTS=(8000 8001 8002 8003)
GPUS=(0 1 2 3)
GPU_UTIL=0.90
MAX_MODEL_LEN=24576    # visual: 5 skill-card images + SkillNet prompts + agent history

if [ ! -x "$PYTHON" ]; then
    echo "[ERROR] $PYTHON not found."
    exit 1
fi

PID_FILE="$SCRIPT_DIR/vllm_pids.txt"
: > "$PID_FILE"

echo "Starting ${#PORTS[@]} vLLM instances for $MODEL ..."
for i in "${!PORTS[@]}"; do
    PORT=${PORTS[$i]}
    GPU=${GPUS[$i]}
    LOG="$SCRIPT_DIR/vllm_${PORT}.log"
    echo "  GPU $GPU -> port $PORT (log: $LOG)"
    CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --port "$PORT" \
        --tensor-parallel-size 1 \
        --max-model-len "$MAX_MODEL_LEN" \
        --gpu-memory-utilization "$GPU_UTIL" \
        --no-enable-log-requests \
        --limit-mm-per-prompt '{"image": 6}' \
        > "$LOG" 2>&1 &
    echo $! >> "$PID_FILE"
    echo "  PID: $!"
done

echo ""
echo "Waiting for servers to be ready (first startup loads weights, be patient)..."
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
