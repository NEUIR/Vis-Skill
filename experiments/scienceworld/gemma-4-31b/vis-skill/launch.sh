#!/bin/bash
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
# ============================================================================
# ScienceWorld vis-skill CONTROLLED (Gemma-4-31B) — evaluation launcher.
#
# Controlled counterpart of ../text-skill: uses EXACTLY the same per-episode
# controlled selections (qwen3.8-max pre-annotation top-5, model-independent),
# rendered as skill-card images (renderer v6, pre-rendered, shared via
# ../shared/skill_cards_v6). Reference flow:
# ../../../qwen3.5-9b/vis-skill.
#
# skill_selections.json lives in ../shared/ and is produced by
# shared/code/prepare_skill_selections.py (hard-gated in ../run_all.sh).
#
# Dynamic scheduling: ALL episodes sit in ONE shared queue; 32 persistent
# workers (multimodal prompts are heavier) each pull the NEXT episode the
# moment they finish their current one, and every LLM request round-robins
# across BOTH 2 online multimodal vLLM servers (ports 8000-8001).
#
# The runner code lives in ../shared/code/ (single copy for all three
# experiments); datasets are read from ../shared/ (symlinks).
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-$(command -v python)}"
MODEL="${MODELS_DIR:-$REPO_ROOT/models}/Gemma-4-31B-it"
MODEL_NAME="Gemma-4-31B-it"
SHARED_DIR="$SCRIPT_DIR/../shared"
BASE_PORT=8000
NUM_PORTS=2
MAX_WORKERS=32       # multimodal prompts are heavier; still high concurrency
MAX_STEPS=30
EPISODE_TIMEOUT=1500 # hard wall-clock limit per episode (seconds)
MAX_ATTEMPTS=2       # attempts per episode before an error result is recorded

RESULTS_DIR="$SCRIPT_DIR/results/scienceworld/$MODEL_NAME/vis_skill"

# Preconditions: controlled selections + pre-rendered skill cards (shared).
if [ ! -f "$SHARED_DIR/skill_selections.json" ]; then
    echo "[ERROR] $SHARED_DIR/skill_selections.json not found."
    echo "        Build it from the qwen3.8-max annotation first:"
    echo "        $PYTHON $SHARED_DIR/code/prepare_skill_selections.py \\"
    echo "            --annotations $SHARED_DIR/science_skill_annotations.json \\"
    echo "            --indices $SHARED_DIR/test_indices.json \\"
    echo "            --skills-dir $SHARED_DIR/skills_scienceworld \\"
    echo "            --out $SHARED_DIR/skill_selections.json"
    exit 1
fi
if [ "$(find -L "$SHARED_DIR/skill_cards_v6" -name '*.png' 2>/dev/null | wc -l)" -eq 0 ]; then
    echo "[ERROR] no pre-rendered skill cards under $SHARED_DIR/skill_cards_v6 —"
    echo "        run: bash $REPO_ROOT/scripts/render_skill_cards.sh"
    exit 1
fi

# Standalone convenience: check the shared online vLLM servers (started by
# ../start_vllm.sh WITH the multimodal flags; reused by all three 31B runs).
for i in $(seq 0 $((NUM_PORTS - 1))); do
    PORT=$((BASE_PORT + i))
    if ! curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "[launch] vLLM on port $PORT not healthy — starting the shared servers..."
        bash "$SCRIPT_DIR/../start_vllm.sh"
        break
    fi
done

"$PYTHON" "$SHARED_DIR/code/run_scienceworld_vis.py" \
    --model "$MODEL" \
    --use_skill \
    --output_path "$RESULTS_DIR" \
    --experiment_name "vis-skill-gemma-31B-scienceworld" \
    --max_workers "$MAX_WORKERS" \
    --max_steps "$MAX_STEPS" \
    --base_port "$BASE_PORT" \
    --num_ports "$NUM_PORTS" \
    --episode_timeout "$EPISODE_TIMEOUT" \
    --max_attempts "$MAX_ATTEMPTS" \
    2>&1 | tee "$SCRIPT_DIR/run.log"

echo "vis-skill controlled (ScienceWorld, Gemma-4-31B) done. Log: $SCRIPT_DIR/run.log"
echo "Results: $RESULTS_DIR"
