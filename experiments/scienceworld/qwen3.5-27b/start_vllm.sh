#!/bin/bash
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
# ============================================================================
# Shared online vLLM server for ScienceWorld 27B experiments.
#
# Launches 2 independent vLLM processes (tensor-parallel-size=2, each using
# 2 GPUs, ports 8000-8001). CUDA graph enabled by default for acceleration.
# Client workers round-robin across both ports for load balancing.
# ============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"

MODEL="${MODELS_DIR:-$REPO_ROOT/models}/Qwen3.5-27B"
PYTHON="${PYTHON:-$(command -v python)}"
BASE_PORT=8000
NUM_INSTANCES=2
TP_SIZE=2
LOG_PREFIX="$ROOT/vllm_server"
PID_FILE="$ROOT/vllm_server.pid"

# Check if already running — and serving the RIGHT model (ports occupied by
# another backbone must not be silently reused).
if curl -s "http://localhost:$BASE_PORT/health" > /dev/null 2>&1; then
    if curl -s "http://localhost:$BASE_PORT/v1/models" | grep -q "$MODEL"; then
        echo "[start_vllm] vLLM already healthy on port $BASE_PORT and serving $MODEL, reusing it."
        exit 0
    fi
    echo "[start_vllm][ERROR] Port $BASE_PORT is healthy but serving a DIFFERENT model:"
    curl -s "http://localhost:$BASE_PORT/v1/models" | head -c 300; echo
    echo "[start_vllm][ERROR] Free ports ${BASE_PORT}-$((BASE_PORT + NUM_INSTANCES - 1)) first (stop the other vLLM servers)."
    exit 1
fi

echo "[start_vllm] Starting $NUM_INSTANCES vLLM servers (TP=$TP_SIZE each, ports $BASE_PORT-$((BASE_PORT + NUM_INSTANCES - 1)))"
echo "[start_vllm] Model: $MODEL, gpu-util=0.90, max-model-len=32768, cuda-graph enabled (default)"

> "$PID_FILE"

for i in $(seq 0 $((NUM_INSTANCES - 1))); do
    PORT=$((BASE_PORT + i))
    # Each instance uses 2 consecutive GPUs
    GPU_START=$((i * TP_SIZE))
    GPU_END=$((GPU_START + TP_SIZE - 1))
    GPU_IDS=$(seq -s, $GPU_START $GPU_END)

    CUDA_VISIBLE_DEVICES=$GPU_IDS setsid "$PYTHON" -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --host 0.0.0.0 \
        --port "$PORT" \
        --tensor-parallel-size "$TP_SIZE" \
        --gpu-memory-utilization 0.90 \
        --max-model-len 32768 \
        --no-enable-log-requests \
        --trust-remote-code \
        --limit-mm-per-prompt '{"image": 8}' \
        > "${LOG_PREFIX}_${PORT}.log" 2>&1 &
    echo $! >> "$PID_FILE"
    echo "[start_vllm] GPUs $GPU_IDS -> port $PORT (PID $!)"
done

# Wait for all servers to be healthy
MAX_WAIT=1200
WAITED=0
while [ "$WAITED" -lt "$MAX_WAIT" ]; do
    ALL_HEALTHY=true
    for i in $(seq 0 $((NUM_INSTANCES - 1))); do
        PORT=$((BASE_PORT + i))
        if ! curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
            ALL_HEALTHY=false
            break
        fi
    done
    if [ "$ALL_HEALTHY" = true ]; then
        echo "[start_vllm] All $NUM_INSTANCES servers ready (ports $BASE_PORT-$((BASE_PORT + NUM_INSTANCES - 1))) after ${WAITED}s."
        exit 0
    fi
    sleep 5
    WAITED=$((WAITED + 5))
done

echo "[start_vllm][ERROR] Timed out after ${MAX_WAIT}s. Last log lines:"
for i in $(seq 0 $((NUM_INSTANCES - 1))); do
    PORT=$((BASE_PORT + i))
    echo "--- port $PORT ---"
    tail -10 "${LOG_PREFIX}_${PORT}.log" 2>/dev/null
done
exit 1
