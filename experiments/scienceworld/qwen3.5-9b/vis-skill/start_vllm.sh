#!/bin/bash
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
# ============================================================================
# Shared online vLLM server for ALL three ScienceWorld 9B experiments.
#
# Launches 4 independent vLLM processes (one per GPU, ports 8000-8003).
# Client workers round-robin across all 4 ports for load balancing.
# ============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"

MODEL="${MODELS_DIR:-$REPO_ROOT/models}/Qwen3.5-9B"
PYTHON="${PYTHON:-$(command -v python)}"
BASE_PORT=8000
NUM_GPUS=4
LOG_PREFIX="$ROOT/vllm_server"
PID_FILE="$ROOT/vllm_server.pid"

# Check if already running — and serving the RIGHT model (ports occupied by
# another backbone, e.g. the 27B experiment, must not be silently reused).
if curl -s "http://localhost:$BASE_PORT/health" > /dev/null 2>&1; then
    if curl -s "http://localhost:$BASE_PORT/v1/models" | grep -q "$MODEL"; then
        echo "[start_vllm] vLLM already healthy on port $BASE_PORT and serving $MODEL, reusing it."
        exit 0
    fi
    echo "[start_vllm][ERROR] Port $BASE_PORT is healthy but serving a DIFFERENT model:"
    curl -s "http://localhost:$BASE_PORT/v1/models" | head -c 300; echo
    echo "[start_vllm][ERROR] Free ports ${BASE_PORT}-$((BASE_PORT + NUM_GPUS - 1)) first (stop the other vLLM servers)."
    exit 1
fi

echo "[start_vllm] Starting 4 independent vLLM servers (one per GPU, ports 8000-8003)"
echo "[start_vllm] Model: $MODEL, gpu-util=0.80, max-model-len=32768, cuda-graph (vllm 0.26.0)"

> "$PID_FILE"

for i in $(seq 0 $((NUM_GPUS - 1))); do
    PORT=$((BASE_PORT + i))
    CUDA_VISIBLE_DEVICES=$i setsid "$PYTHON" -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --host 0.0.0.0 \
        --port "$PORT" \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.80 \
        --max-model-len 32768 \
        --no-enable-log-requests \
        --trust-remote-code \
        --limit-mm-per-prompt '{"image": 8}' \
        > "${LOG_PREFIX}_${PORT}.log" 2>&1 &
    echo $! >> "$PID_FILE"
    echo "[start_vllm] GPU $i -> port $PORT (PID $!)"
done

# Wait for all servers to be healthy
MAX_WAIT=900
WAITED=0
while [ "$WAITED" -lt "$MAX_WAIT" ]; do
    ALL_HEALTHY=true
    for i in $(seq 0 $((NUM_GPUS - 1))); do
        PORT=$((BASE_PORT + i))
        if ! curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
            ALL_HEALTHY=false
            break
        fi
    done
    if [ "$ALL_HEALTHY" = true ]; then
        echo "[start_vllm] All 4 servers ready (ports 8000-8003) after ${WAITED}s."
        exit 0
    fi
    sleep 5
    WAITED=$((WAITED + 5))
done

echo "[start_vllm][ERROR] Timed out after ${MAX_WAIT}s. Last log lines:"
for i in $(seq 0 $((NUM_GPUS - 1))); do
    PORT=$((BASE_PORT + i))
    echo "--- port $PORT ---"
    tail -10 "${LOG_PREFIX}_${PORT}.log" 2>/dev/null
done
exit 1
