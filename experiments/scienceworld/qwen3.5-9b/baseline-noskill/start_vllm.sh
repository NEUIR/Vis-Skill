#!/bin/bash
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
# ============================================================================
# Start ONE online vLLM server for Qwen3.5-9B (ScienceWorld experiments).
#
# Dynamic scheduling design:
#   --data-parallel-size 4 puts one DP rank on each of the 4 GPUs behind a
#   SINGLE endpoint (port 8000). vLLM's frontend routes every incoming
#   request to the currently least-loaded rank, so requests are balanced
#   dynamically at runtime and all GPUs stay busy until the last request.
#
# ../run_all.sh calls launch.sh (which starts this server) and stops it with
# stop_vllm.sh before the shared deployment for the skill runs is started.
# ============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"

MODEL="${MODELS_DIR:-$REPO_ROOT/models}/Qwen3.5-9B"
PYTHON="${PYTHON:-$(command -v python)}"
PORT=8000
LOG="$ROOT/vllm_${PORT}.log"
PID_FILE="$ROOT/vllm_${PORT}.pid"

if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
    echo "[start_vllm] vLLM already healthy on port $PORT, reusing it."
    exit 0
fi

echo "[start_vllm] Starting vLLM: $MODEL"
echo "[start_vllm] DP=4 (one rank per GPU), gpu-util=0.9, max-model-len=65536, port=$PORT"

CUDA_VISIBLE_DEVICES=0,1,2,3 setsid "$PYTHON" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --tensor-parallel-size 1 \
    --data-parallel-size 4 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 65536 \
    --no-enable-log-requests \
    --trust-remote-code \
    --limit-mm-per-prompt '{"image": 8}' \
    > "$LOG" 2>&1 &
echo $! > "$PID_FILE"
echo "[start_vllm] PID $(cat "$PID_FILE"), log: $LOG"

MAX_WAIT=900
WAITED=0
while [ "$WAITED" -lt "$MAX_WAIT" ]; do
    if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "[start_vllm] Server ready on port $PORT after ${WAITED}s."
        exit 0
    fi
    if ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        echo "[start_vllm][ERROR] vLLM process died during startup. Last log lines:"
        tail -30 "$LOG"
        exit 1
    fi
    sleep 5
    WAITED=$((WAITED + 5))
done

echo "[start_vllm][ERROR] Timed out after ${MAX_WAIT}s waiting for vLLM. Last log lines:"
tail -30 "$LOG"
exit 1
