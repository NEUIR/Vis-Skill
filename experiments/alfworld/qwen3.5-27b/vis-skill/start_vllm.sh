#!/bin/bash
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
# =============================================================================
# Start ONLINE vLLM servers for Qwen3.5-27B (multimodal) — 2 replicas,
# tensor-parallel=2.
#
#   - GPU(0,1) -> port 8000, GPU(2,3) -> port 8001
#   - Rationale: 27B weights (~52GB bf16) do not leave enough KV-cache
#     headroom on a single 80GB GPU for high concurrency, so TP=2 per
#     replica (~26GB weights/GPU, ~46GB KV-cache/GPU); 2 independent
#     replicas avoid a single scheduler becoming a bottleneck and match
#     run_visual.py's multi-port round-robin load-balancing (--ports).
#   - serving flags: max-model-len 24576, gpu-util 0.90,
#     limit-mm-per-prompt image:6 (5 skill-card images + headroom)
#   - clients (code/run_visual.py) disable thinking via
#     chat_template_kwargs.enable_thinking=false
#
# If servers from a previous experiment occupy these ports/GPUs, stop them
# first (bash stop_vllm.sh there), they are independent processes.
# =============================================================================
set -u

MODEL="${MODELS_DIR:-$REPO_ROOT/models}/Qwen3.5-27B"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-$(command -v python)}"

REPLICAS=(
    "0,1:8000"
    "2,3:8001"
)
GPU_UTIL=0.90
MAX_MODEL_LEN=24576    # visual: 5 skill-card images + SkillNet prompts + agent history

if [ ! -x "$PYTHON" ]; then
    echo "[ERROR] $PYTHON not found."
    exit 1
fi

PID_FILE="$SCRIPT_DIR/vllm_pids.txt"
: > "$PID_FILE"

echo "Starting ${#REPLICAS[@]} vLLM TP=2 instances for $MODEL ..."
for spec in "${REPLICAS[@]}"; do
    GPUS="${spec%%:*}"
    PORT="${spec##*:}"
    LOG="$SCRIPT_DIR/vllm_${PORT}.log"
    echo "  GPUs $GPUS -> port $PORT (log: $LOG)"
    CUDA_VISIBLE_DEVICES=$GPUS "$PYTHON" -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --port "$PORT" \
        --tensor-parallel-size 2 \
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
for spec in "${REPLICAS[@]}"; do
    PORT="${spec##*:}"
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
