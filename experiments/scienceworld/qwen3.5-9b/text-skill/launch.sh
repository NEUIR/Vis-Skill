#!/bin/bash
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
# ============================================================================
# ScienceWorld text-skill CONTROLLED (Qwen3.5-9B) — evaluation launcher.
#
# SkillNet 3-stage skill pipeline (SKILL.md only, no reference files), but
# stage-1 skill retrieval is FIXED to the pre-annotated TOP-5 most relevant
# skills per episode from science_skill_annotations.json (annotated offline
# with qwen3.8-max). Stages 2/3 are unchanged.
#
# Skill selections are recorded per episode and exported to
# skill_selections.json when all 211 episodes are done (cross-checked by the
# vis-skill experiment's prepare_skill_selections.py).
#
# Inherited v1 properties:
#   - ONE shared queue; every worker pulls the NEXT episode as soon as it
#     finishes its current one (true dynamic scheduling)
#   - LLM requests round-robin across ALL 4 online vLLM servers (8000-8003)
#   - hard per-episode wall-clock timeout + process-tree kill + retries
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-$(command -v python)}"
MODEL="${MODELS_DIR:-$REPO_ROOT/models}/Qwen3.5-9B"
BASE_PORT=8000
NUM_PORTS=4
MAX_WORKERS=40       # shared-queue workers; high concurrency, each pulls next when free
MAX_STEPS=30
EPISODE_TIMEOUT=2400 # hard wall-clock limit per episode (seconds)
MAX_ATTEMPTS=2       # attempts per episode before an error result is recorded

# Preconditions: annotation file + skill library must be present (copied,
# never symlinked).
[ -f "$SCRIPT_DIR/code/science_skill_annotations.json" ] || { \
    echo "[ERROR] code/science_skill_annotations.json missing."; exit 1; }
[ -d "$SCRIPT_DIR/skills_scienceworld" ] || { \
    echo "[ERROR] skills_scienceworld/ missing."; exit 1; }

# Standalone convenience: start the shared online vLLM servers if needed.
ALL_HEALTHY=true
for i in $(seq 0 $((NUM_PORTS - 1))); do
    if ! curl -s "http://localhost:$((BASE_PORT + i))/health" > /dev/null 2>&1; then
        ALL_HEALTHY=false; break
    fi
done
if [ "$ALL_HEALTHY" = false ]; then
    bash "$SCRIPT_DIR/start_vllm.sh"
fi

"$PYTHON" code/run_scienceworld_text.py \
    --model "$MODEL" \
    --use_skill \
    --max_workers "$MAX_WORKERS" \
    --max_steps "$MAX_STEPS" \
    --base_port "$BASE_PORT" \
    --num_ports "$NUM_PORTS" \
    --episode_timeout "$EPISODE_TIMEOUT" \
    --max_attempts "$MAX_ATTEMPTS" \
    2>&1 | tee "$SCRIPT_DIR/run.log"

echo "text-skill controlled (ScienceWorld) done. Log: $SCRIPT_DIR/run.log"
