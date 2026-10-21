#!/bin/bash
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
# ============================================================================
# launch.sh — WebShop vis-skill evaluation (Qwen3.5-27B, multimodal)
# v1 fix: ASIN click format correction in system prompt, stage-2 prompt,
#         procedure examples, and code template (stuck detection).
#
# Pipeline (SkillNet 3-stage, visual variant):
#   1. Fixed skill selections from QWEN3.8-MAX ANNOTATIONS (top-5 most relevant
#      skills per instance, webshop_skill_annotations.json).
#      NO model-based skill selection in this experiment.
#   2. VLM reads rendered skill card IMAGES -> generates procedural guidance
#   3. LLM generates Python code (text-only) -> exec() runs agent loop
#
# vLLM: online deployment (ports 8000-8003), TRUE dynamic scheduling:
#       all games live in one shared queue; a worker that finishes a game
#       immediately pulls the next one, so all GPUs stay busy to the end.
# Backbone: models/Qwen3.5-27B
# Config:
#   --max-model-len 65536, --gpu-memory-utilization 0.85, max_tokens uncapped
# ============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_PATH="${MODELS_DIR:-$REPO_ROOT/models}/Qwen3.5-27B"
PYTHON="${PYTHON:-$(command -v python)}"
NUM_GPUS=4
MAX_STEPS=30
# NOTE: items_shuffle.json is ~5.5GB and is parsed by EVERY worker
# (~14GB RSS each). 12 workers is the proven-safe max on the 256GB pod
# (16 workers exceeded the memcg limit during concurrent env load).
MAX_WORKERS=12
EXP_NAME="qwen27b_webshop_visskill_v1"
TOTAL_GAMES=200

export API_KEY=EMPTY
export JAVA_HOME="${JAVA_HOME:-$("${PYTHON:-python}" -c 'import jdk4py; print(jdk4py.JAVA_HOME)')}"

echo "======================================================"
echo "  WebShop Vis-Skill-v1 — Qwen3.5-27B (ASIN click fix)"
echo "======================================================"
echo "  Model:       $MODEL_PATH"
echo "  GPUs:        $NUM_GPUS"
echo "  Workers:     $MAX_WORKERS (distributed across ports)"
echo "  Total Games: $TOTAL_GAMES"
echo "  MAX_STEPS:   $MAX_STEPS"
echo "  Exp Name:    $EXP_NAME"
echo "  Skills:      visual v7 (images/skill_cards_v7, pre-rendered)"
echo "  Selections:  Qwen3.8-Max annotations, top-5 per instance"
echo "======================================================"

# ── Step 0: Build skill selections from Qwen3.8-Max annotations (no model) ──
echo "=== Generating skill_selections.json (annotation top-5) ==="
$PYTHON "$SCRIPT_DIR/code/generate_skill_selections.py"
echo ""

# ── Step 1: Render skill cards (if not already done) ──
# Cards are already rendered in images/skill_cards_v7; this check only
# re-renders if some are missing.
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
# Refuse to run against foreign servers that occupy the required ports
# with a different model (would silently contaminate results).
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
