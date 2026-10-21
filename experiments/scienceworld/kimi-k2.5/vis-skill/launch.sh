#!/bin/bash
# ============================================================================
# ScienceWorld vis-skill CONTROLLED (kimi-k2.5) — evaluation launcher.
#
# Controlled counterpart of ../text-skill: uses EXACTLY the same per-episode
# controlled selections (qwen3.8-max pre-annotation top-5, model-independent),
# rendered as skill-card images (renderer v6, pre-rendered, shared via
# ../shared/skill_cards_v6) and sent to the API as base64 data URLs.
# Flow reference: scienceworld/gemma-4-31b (shared runner); endpoint reference:
# alfworld/kimi-k2.5 — DashScope OpenAI-compatible API, NO local vLLM.
#
# skill_selections.json lives in ../shared/ and is produced by
# shared/code/prepare_skill_selections.py (hard-gated in ../run_all.sh).
#
# The API key is read from $DASHSCOPE_API_KEY (export it before running;
# the base URL can be overridden via $DASHSCOPE_BASE_URL).
#
# Dynamic scheduling: ALL episodes sit in ONE shared queue; 20 persistent
# workers (multimodal prompts are heavier) each pull the NEXT episode the
# moment they finish their current one against the shared API endpoint.
#
# The runner code lives in ../shared/code/ (single copy for all three
# experiments); datasets are read from ../shared/ (symlinks).
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-$(command -v python)}"
MODEL="kimi-k2.5"
MODEL_NAME="kimi-k2.5"
SHARED_DIR="$SCRIPT_DIR/../shared"
MAX_WORKERS=20       # multimodal prompts are heavier; still 20-way concurrency
MAX_STEPS=30
TEMPERATURE=0.0      # SkillNet protocol: temperature=0 on the request side
EPISODE_TIMEOUT=1800 # hard wall-clock limit per episode (seconds)
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
    echo "        run: bash scripts/render_skill_cards.sh (from the repo root)"
    exit 1
fi

# --- DashScope API key check ---
if [ -z "${DASHSCOPE_API_KEY:-}" ]; then
    echo "[ERROR] DASHSCOPE_API_KEY is not set."
    echo "        Export your own DashScope API key first, e.g.:"
    echo "        export DASHSCOPE_API_KEY=<your-key>"
    exit 1
fi

# quick API connectivity check
BASE_URL="${DASHSCOPE_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
    -H "Authorization: Bearer $DASHSCOPE_API_KEY" \
    "$BASE_URL/models")
if [ "$HTTP_CODE" != "200" ]; then
    echo "[WARN] DashScope API check returned HTTP $HTTP_CODE ($BASE_URL/models)."
    echo "       Continuing anyway; if the key/network is bad the run will fail."
fi

"$PYTHON" "$SHARED_DIR/code/run_scienceworld_vis.py" \
    --model "$MODEL" \
    --use_skill \
    --output_path "$RESULTS_DIR" \
    --experiment_name "vis-skill-kimi-k2.5-scienceworld" \
    --max_workers "$MAX_WORKERS" \
    --max_steps "$MAX_STEPS" \
    --temperature "$TEMPERATURE" \
    --episode_timeout "$EPISODE_TIMEOUT" \
    --max_attempts "$MAX_ATTEMPTS" \
    2>&1 | tee "$SCRIPT_DIR/run.log"

echo "vis-skill controlled (ScienceWorld, $MODEL, DashScope API) done. Log: $SCRIPT_DIR/run.log"
echo "Results: $RESULTS_DIR"
