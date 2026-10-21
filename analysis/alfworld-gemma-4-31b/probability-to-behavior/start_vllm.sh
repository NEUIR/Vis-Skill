#!/bin/bash
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
# vLLM for P2B-gemma. 2×TP=2 layout, same as the other probe stages. Reuse via SKIP_VLLM=1.
set -u
MODEL="${GEMMA_MODEL:-${MODELS_DIR:-$REPO_ROOT/models}/Gemma-4-31B-it}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${GEMMA_PY:-${PYTHON:-python}}"

PORTS=(8000 8001); GPU_SPLITS=("0,1" "2,3"); TP=2
GPU_UTIL="${GPU_UTIL:-0.80}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
PREFIX_CACHING="${PREFIX_CACHING:-0}"

RUNTIME_DIR="$SCRIPT_DIR/.runtime"
mkdir -p "$RUNTIME_DIR/cache" "$RUNTIME_DIR/inductor" "$RUNTIME_DIR/triton" "$RUNTIME_DIR/home"
export HOME="${P2B_HOME_OVERRIDE:-$RUNTIME_DIR/home}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$RUNTIME_DIR/cache}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$RUNTIME_DIR/cache/vllm}"
export FLASHINFER_WORKSPACE_DIR="${FLASHINFER_WORKSPACE_DIR:-$RUNTIME_DIR/cache/flashinfer}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$RUNTIME_DIR/inductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$RUNTIME_DIR/triton}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$RUNTIME_DIR/cache/nv}"
[ -z "${TMPDIR:-}" ] && export TMPDIR=/tmp

EXTRA_ARGS=()
[ "$PREFIX_CACHING" = "1" ] && EXTRA_ARGS+=(--enable-prefix-caching)

PID_FILE="$SCRIPT_DIR/vllm_pids.txt"; : > "$PID_FILE"
echo "Starting vLLM for $MODEL (TP=$TP × ${#PORTS[@]}) ..."
for i in "${!PORTS[@]}"; do
    PORT=${PORTS[$i]}; GPUS=${GPU_SPLITS[$i]}
    LOG="$SCRIPT_DIR/vllm_${PORT}.log"
    CUDA_VISIBLE_DEVICES=$GPUS "$PYTHON" -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --port "$PORT" --tensor-parallel-size "$TP" \
        --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization "$GPU_UTIL" \
        --dtype bfloat16 --limit-mm-per-prompt '{"image": 6}' \
        --no-enable-log-requests \
        ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} > "$LOG" 2>&1 &
    echo "$!" >> "$PID_FILE"
    echo "  GPUs $GPUS -> port $PORT  (log: $LOG)"
    sleep 2
done

for PORT in "${PORTS[@]}"; do
    for _ in $(seq 1 240); do
        [ "$(curl -s -o /dev/null -w '%{http_code}' -m 3 http://localhost:$PORT/health || true)" = "200" ] && break
        sleep 5
    done
done
