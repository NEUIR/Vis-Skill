#!/bin/bash
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
# =============================================================================
# Run the TEXT-SKILL experiment (Gemma-4-31B, SKILL.md only).
# CONTROLLED variant — flow identical to
# experiments/webshop/qwen3.5-9b/text-skill.
# Prerequisite: the online vLLM servers are up (bash start_vllm.sh).
#
# Pipeline (2-stage, controlled):
#   1. [SKIP] model-based skill retrieval — replaced by the pre-annotated
#      TOP-5 skills per game from webshop_skill_annotations.json
#   2. the model reads the SKILL.md text of those skills -> phased
#      procedural guidance (reference files are NEVER read — SKILL.md only)
#   3. the model generates Python code -> exec() drives the agent loop
#
# Deployment: 2 ONLINE vLLM instances x TP=2 (GPUs 0,1 -> port 8000,
# GPUs 2,3 -> port 8001).
#
# Dynamic scheduling: 12 worker processes pull games from ONE shared queue;
# whenever a worker finishes a game it immediately pulls the next one, so
# all 4 GPUs stay busy until the last game finishes.
#
# The vis-skill experiment derives the SAME top-5 selections from the same
# annotation file (vis-skill/code/generate_skill_selections.py), so both
# experiments use identical skills per task by construction.
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

# quick health check of the vLLM servers
for PORT in 8000 8001; do
    if ! curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "[ERROR] vLLM server on port $PORT is not healthy. Run: bash start_vllm.sh"
        exit 1
    fi
done

ANNOTATIONS="$REPO_ROOT/annotation/outputs/webshop_skill_annotations.json"
if [ ! -f "$ANNOTATIONS" ]; then
    echo "[ERROR] annotations file not found: $ANNOTATIONS"
    exit 1
fi

"$PYTHON" "$SCRIPT_DIR/run_gemma_textskill.py" \
    --model ${MODELS_DIR:-$REPO_ROOT/models}/Gemma-4-31B-it \
    --use_skill \
    --annotations "$ANNOTATIONS" \
    --max_workers 12 \
    --max_steps 30 \
    --exp_name gemma_webshop_textskill \
    --ports "8000,8001" \
    2>&1 | tee "$SCRIPT_DIR/logs_run.log"

echo "Done. Results: results/Gemma-4-31B-it/gemma_webshop_textskill_skill_True/"
