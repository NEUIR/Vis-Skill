#!/bin/bash
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
# ============================================================================
# ScienceWorld text-skill CONTROLLED (Gemma-4-31B) — evaluation launcher.
#
# SkillNet 3-stage skill pipeline (SKILL.md only, no reference files);
# stage-1 skill retrieval is FIXED to the pre-annotated TOP-5 most relevant
# skills per episode from the qwen3.8-max annotation (model-independent,
# shared via ../shared/). Reference flow:
# ../../../qwen3.5-9b/text-skill.
#
# Dynamic scheduling: ALL episodes sit in ONE shared queue; 48 persistent
# workers each pull the NEXT episode the moment they finish their current
# one, and every LLM request round-robins across BOTH 2 online vLLM servers
# (ports 8000-8001) — all GPUs stay busy until the last task.
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
MAX_WORKERS=48       # shared-queue workers; high concurrency, each pulls next when free
MAX_STEPS=30
EPISODE_TIMEOUT=2400 # hard wall-clock limit per episode (seconds)
MAX_ATTEMPTS=2       # attempts per episode before an error result is recorded

RESULTS_DIR="$SCRIPT_DIR/results/scienceworld/$MODEL_NAME/text_skill"

# Preconditions: shared datasets (annotation + skill library).
[ -f "$SHARED_DIR/science_skill_annotations.json" ] || { \
    echo "[ERROR] shared/science_skill_annotations.json missing — run: bash $SHARED_DIR/setup_shared.sh"; exit 1; }
[ -d "$SHARED_DIR/skills_scienceworld" ] || { \
    echo "[ERROR] shared/skills_scienceworld missing — run: bash $SHARED_DIR/setup_shared.sh"; exit 1; }

# Standalone convenience: check the shared online vLLM servers (started by
# ../start_vllm.sh, reused by all three 31B experiments).
for i in $(seq 0 $((NUM_PORTS - 1))); do
    PORT=$((BASE_PORT + i))
    if ! curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "[launch] vLLM on port $PORT not healthy — starting the shared servers..."
        bash "$SCRIPT_DIR/../start_vllm.sh"
        break
    fi
done

"$PYTHON" "$SHARED_DIR/code/run_scienceworld_text.py" \
    --model "$MODEL" \
    --use_skill \
    --output_path "$RESULTS_DIR" \
    --experiment_name "text-skill-gemma-31B-scienceworld" \
    --selections_export "$SCRIPT_DIR/skill_selections_text_run.json" \
    --max_workers "$MAX_WORKERS" \
    --max_steps "$MAX_STEPS" \
    --base_port "$BASE_PORT" \
    --num_ports "$NUM_PORTS" \
    --episode_timeout "$EPISODE_TIMEOUT" \
    --max_attempts "$MAX_ATTEMPTS" \
    2>&1 | tee "$SCRIPT_DIR/run.log"

echo "text-skill controlled (ScienceWorld, Gemma-4-31B) done. Log: $SCRIPT_DIR/run.log"
echo "Results: $RESULTS_DIR"
