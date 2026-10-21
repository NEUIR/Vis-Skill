#!/bin/bash
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
# ============================================================================
# ScienceWorld vis-skill CONTROLLED (Qwen3.5-27B) — evaluation launcher.
#
# Controlled counterpart of ../text-skill: uses EXACTLY the same per-episode
# controlled selections (qwen3.8-max pre-annotation top-5), rendered as
# skill-card images (renderer v6, cards pre-rendered in images/skill_cards_v6/).
#
# skill_selections.json is produced by code/prepare_skill_selections.py from
# science_skill_annotations.json (hard-gated: complete annotation + library
# membership + optional cross-check against the text-skill results).
#
# vLLM deployment: 2 ONLINE instances with tensor-parallel-size=2 (each uses
# 2 GPUs, ports 8000-8001), CUDA graph enabled (no --enforce-eager).
#
# Inherited v1 properties: ONE shared queue (true dynamic scheduling — a
# worker pulls the NEXT episode the moment it finishes its current one), LLM
# requests round-robin across ALL online vLLM servers, hard per-episode
# wall-clock timeout + process-tree kill + retries.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-$(command -v python)}"
MODEL="${MODELS_DIR:-$REPO_ROOT/models}/Qwen3.5-27B"
BASE_PORT=8000
NUM_PORTS=2
MAX_WORKERS=32       # multimodal prompts are heavier; 32 shared-queue workers
MAX_STEPS=30
EPISODE_TIMEOUT=1200 # hard wall-clock limit per episode (seconds; 2x the 9B
                     # value because the 27B backbone decodes slower)
MAX_ATTEMPTS=2       # attempts per episode before an error result is recorded

# Preconditions: controlled (annotation-derived) skill selections.
if [ ! -f "$SCRIPT_DIR/skill_selections.json" ]; then
    echo "[ERROR] $SCRIPT_DIR/skill_selections.json not found."
    echo "        Build it from the qwen3.8-max annotation first:"
    echo "        $PYTHON code/prepare_skill_selections.py \\"
    echo "            --annotations code/science_skill_annotations.json \\"
    echo "            --indices code/test_indices.json \\"
    echo "            --skills-dir skills_scienceworld \\"
    echo "            --out skill_selections.json"
    exit 1
fi

# Render skill cards if not done yet (idempotent, CPU-only); normally they
# are already rendered by scripts/render_skill_cards.sh.
if [ ! -d "$SCRIPT_DIR/images/skill_cards_v6" ] || \
   [ "$(find "$SCRIPT_DIR/images/skill_cards_v6" -name '*.png' 2>/dev/null | wc -l)" -eq 0 ]; then
    bash "$SCRIPT_DIR/render.sh"
fi

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

"$PYTHON" code/run_scienceworld_vis.py \
    --model "$MODEL" \
    --use_skill \
    --max_workers "$MAX_WORKERS" \
    --max_steps "$MAX_STEPS" \
    --base_port "$BASE_PORT" \
    --num_ports "$NUM_PORTS" \
    --episode_timeout "$EPISODE_TIMEOUT" \
    --max_attempts "$MAX_ATTEMPTS" \
    2>&1 | tee "$SCRIPT_DIR/run.log"

echo "vis-skill controlled (ScienceWorld, Qwen3.5-27B) done. Log: $SCRIPT_DIR/run.log"
