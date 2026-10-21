#!/bin/bash
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
# ============================================================================
# launch.sh — WebShop vis-skill evaluation (Qwen3.5-9B, multimodal)
# Same v1 ASIN-click-fix flow as the Qwen3.5-27B launch.sh; only the backbone is swapped
# (8 workers).
# ============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_PATH="${MODELS_DIR:-$REPO_ROOT/models}/Qwen3.5-9B"
PYTHON="${PYTHON:-$(command -v python)}"
NUM_GPUS=4
MAX_STEPS=30
MAX_WORKERS=8
EXP_NAME="qwen9b_webshop_visskill_v1"
TOTAL_GAMES=200

export API_KEY=EMPTY
export JAVA_HOME="${JAVA_HOME:-$("${PYTHON:-python}" -c 'import jdk4py; print(jdk4py.JAVA_HOME)')}"

echo "======================================================"
echo "  WebShop Vis-Skill-v1 — Qwen3.5-9B (ASIN click fix)"
echo "======================================================"
echo "  Model:       $MODEL_PATH"
echo "  GPUs:        $NUM_GPUS"
echo "  Workers:     $MAX_WORKERS"
echo "  Total Games: $TOTAL_GAMES"
echo "  MAX_STEPS:   $MAX_STEPS"
echo "  Exp Name:    $EXP_NAME"
echo "======================================================"

# ── Step 0: Build skill selections from Qwen3.8-Max annotations (no model) ──
echo "=== Generating skill_selections.json (annotation top-5) ==="
$PYTHON "$SCRIPT_DIR/code/generate_skill_selections.py"
echo ""

# ── Step 1: Render skill cards (if not already done) ──
SKILL_CARDS_DIR="$SCRIPT_DIR/images/skill_cards_v7"
SKILL_COUNT=$(ls "$SCRIPT_DIR/skills/" | wc -l)
CARD_COUNT=$(ls "$SKILL_CARDS_DIR"/*.png 2>/dev/null | wc -l)

if [ "$CARD_COUNT" -lt "$SKILL_COUNT" ]; then
    echo ""
    echo "=== Rendering skill cards (renderer v7 / skill_cards_v7) ==="
    $PYTHON "$SCRIPT_DIR/code/render_v7.py"
    echo ""
fi

# ── Step 2: Check/Start vLLM servers ──
echo "=== Checking vLLM servers (must serve $MODEL_PATH) ==="
NEED_START=false
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
        NEED_START=true
    fi
done

if [ "$FOREIGN" = true ]; then
    echo ""
    echo "ERROR: foreign vLLM server(s) occupy required ports."
    echo "Stop the old servers (kill their processes / free GPUs), then re-run."
    exit 1
fi

if [ "$NEED_START" = true ]; then
    echo ""
    echo "=== Starting vLLM servers (multimodal, controlled config) ==="
    bash "$SCRIPT_DIR/start_vllm.sh"
fi

# ── Step 3: Launch evaluation ──
echo ""
echo "=== Launching evaluation ($MAX_WORKERS workers, 4 ports) ==="

$PYTHON "$SCRIPT_DIR/code/run_visual.py" \
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
