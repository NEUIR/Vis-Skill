#!/bin/bash
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
# =============================================================================
# Run the VIS-SKILL experiment (Gemma-4-31B, rendered skill card images).
# CONTROLLED variant — flow identical to
# experiments/webshop/qwen3.5-9b/vis-skill.
# Prerequisite: the online MULTIMODAL vLLM servers are up (bash start_vllm.sh).
#
# Skill selections come from QWEN3.8-MAX ANNOTATIONS (top-5 most relevant skills
# per game, webshop_skill_annotations.json) — NO model-based skill
# selection. This is the same source + rule the text-skill experiment uses,
# so every task uses exactly the same skills in both experiments.
#
# Step 1 builds skill_selections.json from the annotations (and prints the
# per-task skill usage for confirmation).
# Step 2 runs the evaluation with dynamic scheduling: 12 worker processes
# pull games from ONE shared queue, so all 4 GPUs stay busy until the end.
# Deployment: 2 ONLINE multimodal vLLM instances x TP=2
# (GPUs 0,1 -> port 8000, GPUs 2,3 -> port 8001).
# Skill cards are ALREADY RENDERED (images/skill_cards_v7, 23 PNGs).
# =============================================================================
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-$(command -v python)}"
if [ ! -x "$PYTHON" ]; then
    echo "[ERROR] $PYTHON not found — run: bash ../create_env.sh"
    exit 1
fi

export API_KEY=EMPTY
export JAVA_HOME="${JAVA_HOME:-$("${PYTHON:-python}" -c 'import jdk4py; print(jdk4py.JAVA_HOME)')}"

echo "[step 1] building skill selections from Qwen3.8-Max annotations (top-5) ..."
bash "$SCRIPT_DIR/prepare_skill_selections.sh" || exit 1

echo "[step 2] checking vLLM servers ..."
for PORT in 8000 8001; do
    if ! curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "[ERROR] vLLM server on port $PORT is not healthy. Run: bash start_vllm.sh"
        exit 1
    fi
done

echo "[step 3] running evaluation ..."
"$PYTHON" "$SCRIPT_DIR/code/run_visual.py" \
    --model ${MODELS_DIR:-$REPO_ROOT/models}/Gemma-4-31B-it \
    --use_skill \
    --max_workers 12 \
    --max_steps 30 \
    --exp_name gemma_webshop_visskill \
    --ports "8000,8001" \
    2>&1 | tee "$SCRIPT_DIR/logs_run.log"

echo "Done. Results: results/Gemma-4-31B-it/gemma_webshop_visskill_skill_True/"
