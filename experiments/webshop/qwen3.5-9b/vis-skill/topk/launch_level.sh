#!/bin/bash
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)}"
# ============================================================================
# launch_level.sh — WebShop vis-skill TOP-K, one level
# Usage: bash launch_level.sh <TOP_K> [9b|27b]   (model defaults to 9b)
#   TOP_K: skills (card images) per instance (3, 7, 9, 13, 21)
# Same v1 ASIN-click-fix pipeline as the parent vis-skill; the only
# difference is how many skills are selected per instance.
# ============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TOP_K="${1:?usage: launch_level.sh TOP_K [9b|27b]}"
MODEL_SHORT="${2:-9b}"

case "$MODEL_SHORT" in
    9b)  MODEL_PATH="${MODELS_DIR:-$REPO_ROOT/models}/Qwen3.5-9B";  MAX_MODEL_LEN=49152; MAX_WORKERS=8 ;;
    27b) MODEL_PATH="${MODELS_DIR:-$REPO_ROOT/models}/Qwen3.5-27B"; MAX_MODEL_LEN=65536; MAX_WORKERS=12 ;;
    *) echo "unknown model '$MODEL_SHORT' (use 9b|27b)"; exit 1 ;;
esac

PYTHON="${PYTHON:-$(command -v python)}"
MAX_STEPS=30
EXP_NAME="qwen_webshop_visskill_v1_top${TOP_K}"
MARKER="/tmp/vllm_webshop_v1_topk_mm21_marker.txt"

export API_KEY=EMPTY
export JAVA_HOME="${JAVA_HOME:-$("${PYTHON:-python}" -c 'import jdk4py; print(jdk4py.JAVA_HOME)')}"

echo "======================================================"
echo "  WebShop Vis-Skill-v1 TOP-K — level top-$TOP_K ($MODEL_SHORT)"
echo "======================================================"

# ── Step 0: Build skill selections for this level (annotation top-K) ──
echo "=== Generating skill_selections.json (annotation top-$TOP_K) ==="
$PYTHON "$SCRIPT_DIR/code/generate_skill_selections_topk.py" --top_k "$TOP_K"
echo ""

# ── Check/Start vLLM servers (must be the topk mm21 config for this model) ──
NEED_START=false
FOREIGN=false
for port in 8000 8001 8002 8003; do
    if curl -s "http://localhost:${port}/health" > /dev/null 2>&1; then
        if curl -s "http://localhost:${port}/v1/models" | grep -q "$MODEL_PATH"; then
            echo "  port $port: UP (serving $MODEL_PATH)"
        else
            echo "  port $port: UP but serving a DIFFERENT model"
            FOREIGN=true
        fi
    else
        echo "  port $port: DOWN"
        NEED_START=true
    fi
done

if [ "$FOREIGN" = true ]; then
    echo "Ports serve a different model -> will kill and restart with $MODEL_SHORT."
    NEED_START=true
fi

MARKER_OK=false
[ -f "$MARKER" ] && [ "$(cat "$MARKER")" = "${MODEL_PATH}:${MAX_MODEL_LEN}" ] && MARKER_OK=true

if [ "$NEED_START" = true ] || [ "$MARKER_OK" = false ]; then
    echo "=== (Re)starting vLLM servers (topk config: mm=21, len=$MAX_MODEL_LEN) ==="
    pkill -f "[v]llm.entrypoints" 2>/dev/null || true
    sleep 8
    pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
    sleep 5
    # wait until ports are actually free before starting new servers
    for i in $(seq 1 30); do
        ALL_FREE=true
        for port in 8000 8001 8002 8003; do
            if curl -s -m 2 "http://localhost:${port}/health" > /dev/null 2>&1; then
                ALL_FREE=false
            fi
        done
        if [ "$ALL_FREE" = true ]; then break; fi
        sleep 2
    done
    bash "$SCRIPT_DIR/start_vllm_mm21.sh" "$MODEL_PATH" "$MAX_MODEL_LEN"
fi

# ── Launch evaluation ──
$PYTHON "$SCRIPT_DIR/code/run_visual_topk.py" \
    --model "$MODEL_PATH" \
    --use_skill \
    --top_k "$TOP_K" \
    --max_workers $MAX_WORKERS \
    --max_steps $MAX_STEPS \
    --exp_name "$EXP_NAME" \
    --ports "8000,8001,8002,8003" \
    2>&1 | tee "$SCRIPT_DIR/run_output_${MODEL_SHORT}_top${TOP_K}.log"

echo ""
echo "=== Done: $EXP_NAME ==="
echo "Results: $SCRIPT_DIR/results/$(basename $MODEL_PATH)/${EXP_NAME}_skill_True/"
