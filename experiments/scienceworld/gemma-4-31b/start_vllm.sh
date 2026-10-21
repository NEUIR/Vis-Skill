#!/bin/bash
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
# ============================================================================
# Shared ONLINE vLLM servers for ALL THREE ScienceWorld Gemma-4-31B
# experiments (baseline-noskill / text-skill / vis-skill).
#
# Deployment (matches experiments/alfworld/gemma-4-31b):
#   - 31B bf16 weights are ~62.5GB: one 80GB GPU leaves almost no KV-cache
#     headroom, so each instance uses TP=2 (one instance per GPU PAIR):
#       GPU 0+1 -> port 8000, GPU 2+3 -> port 8001
#   - --gpu-memory-utilization 0.9, --max-model-len 24576
#   - CUDA graph acceleration is ON: vLLM captures CUDA graphs by default;
#     we deliberately do NOT pass --enforce-eager.
#   - Multimodal flags are set for the vis-skill run; they do NOT affect
#     text-only requests, so the SAME servers serve all three experiments:
#       --limit-mm-per-prompt '{"image": 8}'      (stage-2 sends <=5 cards)
#       --mm-processor-kwargs '{"max_soft_tokens": 1120}'
#         (Gemma 4 vision token budget: 70/140/280/560/1120; skill cards are
#          densely rendered text -> highest tier for readability)
#
# Client workers round-robin across both ports for request-level dynamic
# load balancing (episode-level balancing is done by the shared queue).
# The dynamic-scheduling mechanism is IDENTICAL to the other backbones — only the
# port pool is 8000,8001 instead of 8000-8003.
# ============================================================================
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
    echo "[ERROR] $PYTHON not found — run: bash $SCRIPT_DIR/create_env.sh"
    exit 1
fi

# ── Pre-flight: ports must be free OR already serving THIS backbone ──
ALL_HEALTHY=true
for PORT in "${PORTS[@]}"; do
    if ! curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        ALL_HEALTHY=false
        break
    fi
done
if [ "$ALL_HEALTHY" = true ]; then
    WRONG_MODEL=false
    for PORT in "${PORTS[@]}"; do
        if ! curl -s "http://localhost:$PORT/v1/models" | grep -q "Gemma-4-31B-it"; then
            WRONG_MODEL=true
            echo "[start_vllm][ERROR] port $PORT is occupied by a DIFFERENT model:"
            curl -s "http://localhost:$PORT/v1/models" | head -c 300; echo
        fi
    done
    if [ "$WRONG_MODEL" = true ]; then
        echo "[start_vllm][ERROR] Free ports ${PORTS[*]} first (stop the other vLLM servers)."
        exit 1
    fi
    echo "[start_vllm] All servers already healthy and serving $MODEL — reusing them."
    exit 0
fi

PID_FILE="$SCRIPT_DIR/vllm_pids.txt"
: > "$PID_FILE"

echo "[start_vllm] Starting ${#PORTS[@]} ONLINE vLLM instances (TP=$TP) for $MODEL"
echo "[start_vllm] gpu-util=$GPU_UTIL, max-model-len=$MAX_MODEL_LEN,"
echo "[start_vllm] CUDA graphs ON (no --enforce-eager), multimodal flags enabled."
for i in "${!PORTS[@]}"; do
    PORT=${PORTS[$i]}
    GPUS=${GPU_GROUPS[$i]}
    LOG="$SCRIPT_DIR/vllm_${PORT}.log"
    echo "  GPU $GPUS -> port $PORT (log: $LOG)"
    CUDA_VISIBLE_DEVICES=$GPUS setsid "$PYTHON" -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --host 0.0.0.0 \
        --port "$PORT" \
        --tensor-parallel-size "$TP" \
        --max-model-len "$MAX_MODEL_LEN" \
        --gpu-memory-utilization "$GPU_UTIL" \
        --limit-mm-per-prompt '{"image": 8}' \
        --mm-processor-kwargs '{"max_soft_tokens": 1120}' \
        --no-enable-log-requests \
        > "$LOG" 2>&1 &
    echo $! >> "$PID_FILE"
    echo "  PID: $!"
done

echo ""
echo "[start_vllm] Waiting for all servers to be ready (first startup loads ~62.5GB weights)..."
for PORT in "${PORTS[@]}"; do
    WAITED=0
    while ! curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; do
        sleep 5
        WAITED=$((WAITED + 5))
        if [ "$WAITED" -ge 1800 ]; then
            echo "[ERROR] port $PORT not healthy after 1800s — check $SCRIPT_DIR/vllm_${PORT}.log"
            tail -20 "$SCRIPT_DIR/vllm_${PORT}.log" 2>/dev/null
            exit 1
        fi
    done
    echo "  Port $PORT ready."
done
echo "[start_vllm] All vLLM servers are up (CUDA graph acceleration enabled)."
