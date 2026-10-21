#!/bin/bash
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
# ============================================================================
# launch.sh — WebShop text-skill evaluation (Qwen3.5-27B)
# v1 fix: ASIN click format correction in system prompt, stage-2 prompt,
#         procedure examples, and code template (stuck detection).
#
# Pipeline (SkillNet 2-stage, controlled):
#   1. [SKIP] Skill retrieval replaced by pre-annotated top-5 from annotations
#   2. LLM reads SKILL.md text -> generates procedural guidance
#   3. LLM generates Python code -> exec() runs agent loop
#
# vLLM: online deployment, true dynamic scheduling (worker pool pulls from
#        shared queue — all GPUs stay busy until last task)
# Backbone: models/Qwen3.5-27B
# Max model len: 65536
# ============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_PATH="${MODELS_DIR:-$REPO_ROOT/models}/Qwen3.5-27B"
PYTHON="${PYTHON:-$(command -v python)}"
NUM_GPUS=4
MAX_STEPS=30
MAX_WORKERS=12
EXP_NAME="qwen_webshop_textskill_v1"
MAX_MODEL_LEN=65536

export API_KEY=EMPTY
export JAVA_HOME="${JAVA_HOME:-$("${PYTHON:-python}" -c 'import jdk4py; print(jdk4py.JAVA_HOME)')}"

echo "============================================"
echo "  WebShop Text-Skill-v1 — Qwen3.5-27B"
echo "  (ASIN click fix applied)"
echo "============================================"
echo "  Model:         $MODEL_PATH"
echo "  GPUs:          $NUM_GPUS"
echo "  Workers:       $MAX_WORKERS"
echo "  MAX_STEPS:     $MAX_STEPS"
echo "  MAX_MODEL_LEN: $MAX_MODEL_LEN"
echo "  Exp Name:      $EXP_NAME"
echo "============================================"

# ── Check/Start vLLM servers ──
# Refuse to run against foreign servers that occupy the required ports
# with a different model (would silently contaminate results).
echo "=== Checking vLLM servers (must serve $MODEL_PATH) ==="
ALL_UP=true
FOREIGN=false
for port in 8000 8001 8002 8003; do
    if curl -s "http://localhost:${port}/health" > /dev/null 2>&1; then
        if curl -s "http://localhost:${port}/v1/models" | grep -q "$MODEL_PATH"; then
            echo "  port $port: UP (serving $MODEL_PATH)"
        else
            echo "  port $port: UP but serving a DIFFERENT model/config"
            FOREIGN=true
        fi
    else
        echo "  port $port: DOWN"
        ALL_UP=false
    fi
done

if [ "$FOREIGN" = true ]; then
    echo ""
    echo "ERROR: foreign vLLM server(s) occupy required ports."
    echo "Stop the old servers (kill their processes / free GPUs), then re-run."
    exit 1
fi

if [ "$ALL_UP" = false ]; then
    echo ""
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
            --max-model-len $MAX_MODEL_LEN \
            --enforce-eager \
            --no-enable-log-requests \
            > /tmp/vllm_webshop_textskill_v1_gpu${gpu}.log 2>&1 &
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
echo ""
echo "=== Launching evaluation ($MAX_WORKERS workers, 4 ports) ==="

$PYTHON "$SCRIPT_DIR/run_qwen_webshop.py" \
    --model "$MODEL_PATH" \
    --use_skill \
    --max_workers $MAX_WORKERS \
    --max_steps $MAX_STEPS \
    --exp_name "$EXP_NAME" \
    --ports "8000,8001,8002,8003" \
    2>&1 | tee "$SCRIPT_DIR/run_output.log"

echo ""
echo "=== Done ==="
echo "Results: $SCRIPT_DIR/results/$(basename $MODEL_PATH)/${EXP_NAME}_skill_True/"
