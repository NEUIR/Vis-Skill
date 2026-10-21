#!/bin/bash
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_PATH="${MODELS_DIR:-$REPO_ROOT/models}/Qwen3.5-27B"
PYTHON="${PYTHON:-$(command -v python)}"
NUM_GPUS=4
MAX_STEPS=30
MAX_WORKERS=12
EXP_NAME="qwen_webshop_baseline"

export API_KEY=EMPTY
export JAVA_HOME="${JAVA_HOME:-$("${PYTHON:-python}" -c 'import jdk4py; print(jdk4py.JAVA_HOME)')}"

echo "============================================"
echo "  Baseline (No-Skill) WebShop — Qwen3.5-27B"
echo "============================================"
echo "  Model:       $MODEL_PATH"
echo "  GPUs:        $NUM_GPUS"
echo "  Workers:     $MAX_WORKERS (dynamic across 4 ports)"
echo "  Total Games: 200"
echo "  MAX_STEPS:   $MAX_STEPS"
echo "============================================"

# ── Check/Start vLLM servers ──
ALL_UP=true
for port in 8000 8001 8002 8003; do
    if ! curl -s "http://localhost:${port}/health" > /dev/null 2>&1; then
        ALL_UP=false
        break
    fi
done

if [ "$ALL_UP" = false ]; then
    echo "=== Starting vLLM servers ==="
    pkill -f "vllm.entrypoints" 2>/dev/null || true
    sleep 3
    for gpu in $(seq 0 $((NUM_GPUS - 1))); do
        port=$((8000 + gpu))
        echo "  GPU $gpu -> port $port"
        CUDA_VISIBLE_DEVICES=$gpu $PYTHON -m vllm.entrypoints.openai.api_server \
            --model "$MODEL_PATH" \
            --host 0.0.0.0 \
            --port $port \
            --trust-remote-code \
            --tensor-parallel-size 1 \
            --gpu-memory-utilization 0.85 \
            --max-model-len 65536 \
            --enforce-eager \
            --no-enable-log-requests \
            > /tmp/vllm_baseline_27b_gpu${gpu}.log 2>&1 &
    done
    echo "  Waiting for servers..."
    for port in 8000 8001 8002 8003; do
        for i in $(seq 1 120); do
            if curl -s "http://localhost:${port}/health" > /dev/null 2>&1; then
                echo "  port $port ready"
                break
            fi
            sleep 5
        done
    done
fi

# ── Launch evaluation ──
echo "=== Launching evaluation ($MAX_WORKERS workers, 4 ports) ==="
$PYTHON "$SCRIPT_DIR/run_qwen_webshop.py" \
    --model "$MODEL_PATH" \
    --max_workers $MAX_WORKERS \
    --max_steps $MAX_STEPS \
    --exp_name "$EXP_NAME" \
    --ports "8000,8001,8002,8003" \
    2>&1 | tee "$SCRIPT_DIR/run_output.log"

echo "=== Done ==="
