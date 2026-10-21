#!/bin/bash
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
# =============================================================================
# Start ONLINE vLLM servers for Gemma-4-31B — TP=2, 2 instances on 4 GPUs:
#   instance 0: GPUs 0,1 -> port 8000
#   instance 1: GPUs 2,3 -> port 8001
# Baseline (no-skill): text-only serving.
#
# Why TP=2: Gemma-4-31B is ~62.5GB of bf16 weights — a single A100-80GB
# cannot hold weights + 65K-context KV cache, so each instance spans 2 GPUs.
# 2 instances keep all 4 GPUs serving concurrently (higher aggregate
# throughput than one TP=4 instance).
#
#   - gpu-memory-utilization 0.90
#   - max-model-len 65536
#   - the official Gemma 4 chat template from the -it model directory is
#     used (no --tokenizer override)
#   - clients disable thinking via chat_template_kwargs.enable_thinking=false
#
# Port safety: if a port is already serving, this script only reclaims it
# when it belongs to a Gemma-4-31B-it server (leftover from a previous run of
# THIS experiment); servers of other experiments are never touched.
# =============================================================================
set -u

MODEL="${MODELS_DIR:-$REPO_ROOT/models}/Gemma-4-31B-it"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-$(command -v python)}"

PORTS=(8000 8001)
GPU_GROUPS=("0,1" "2,3")
TENSOR_PARALLEL=2
GPU_UTIL=0.90
MAX_MODEL_LEN=65536

if [ ! -x "$PYTHON" ]; then
    echo "[ERROR] $PYTHON not found — run: bash ../create_env.sh"
    exit 1
fi

# ── Weight pre-flight: every shard in the index must be downloaded ──
MISSING_SHARDS=$("$PYTHON" - "$MODEL" <<'PYEOF'
import json, os, sys
model_dir = sys.argv[1]
with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
    index = json.load(f)
shards = sorted(set(index["weight_map"].values()))
missing = [s for s in shards if not os.path.exists(os.path.join(model_dir, s))]
print(" ".join(missing))
PYEOF
)
if [ -n "$MISSING_SHARDS" ]; then
    echo "[ERROR] $MODEL download incomplete — missing shard(s): $MISSING_SHARDS"
    echo "        Wait for the model download to finish, then re-run."
    exit 1
fi

# ── Port pre-flight: reclaim leftover Gemma-4-31B-it servers, never foreign ones ──
for PORT in "${PORTS[@]}"; do
    if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        MODELS=$(curl -s "http://localhost:$PORT/v1/models" 2>/dev/null)
        if echo "$MODELS" | grep -q "Gemma-4-31B-it"; then
            echo "[preflight] port $PORT serves a leftover Gemma-4-31B-it server — stopping it"
            for PID in $(pgrep -f "vllm.entrypoints.openai.api_server.*Gemma-4-31B-it.*--port $PORT" || true); do
                kill "$PID" 2>/dev/null || true
            done
            sleep 15
            if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
                echo "[ERROR] port $PORT still serving after killing leftover Gemma server."
                exit 1
            fi
        else
            echo "[ERROR] port $PORT is occupied by ANOTHER experiment's server."
            echo "        Stop it first (or change PORTS/GPU_GROUPS in this script + --ports in launch.sh)."
            exit 1
        fi
    fi
done

PID_FILE="$SCRIPT_DIR/vllm_pids.txt"
: > "$PID_FILE"

echo "Starting ${#PORTS[@]} vLLM instances (TP=$TENSOR_PARALLEL) for $MODEL (baseline-noskill) ..."
for i in "${!PORTS[@]}"; do
    PORT=${PORTS[$i]}
    GPUS=${GPU_GROUPS[$i]}
    LOG="$SCRIPT_DIR/vllm_${PORT}.log"
    echo "  GPUs $GPUS -> port $PORT (log: $LOG)"
    CUDA_VISIBLE_DEVICES=$GPUS "$PYTHON" -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --host 0.0.0.0 \
        --port "$PORT" \
        --tensor-parallel-size "$TENSOR_PARALLEL" \
        --max-model-len "$MAX_MODEL_LEN" \
        --gpu-memory-utilization "$GPU_UTIL" \
        --no-enable-log-requests \
        > "$LOG" 2>&1 &
    echo $! >> "$PID_FILE"
    echo "  PID: $!"
done

echo ""
echo "Waiting for servers to be ready (first startup loads ~62GB weights per instance, be patient)..."
for PORT in "${PORTS[@]}"; do
    WAITED=0
    while ! curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; do
        sleep 5
        WAITED=$((WAITED + 5))
        if [ "$WAITED" -ge 2400 ]; then
            echo "[ERROR] port $PORT not healthy after 2400s — check $SCRIPT_DIR/vllm_${PORT}.log"
            exit 1
        fi
    done
    echo "  Port $PORT ready."
done
echo "All vLLM servers are up!"
