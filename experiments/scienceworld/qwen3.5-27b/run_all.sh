#!/bin/bash
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
# ============================================================================
# ONE-SHOT launcher for the ScienceWorld 27B CONTROLLED experiments
# (skill selection fixed by the qwen3.8-max pre-annotation, top-5 relevance):
#
#   [1] baseline-noskill -> baseline-noskill/ (its own TP=2 x DP=2 vLLM server on
#       port 8000, stopped afterwards)
#   [2] shared ONLINE vLLM servers (2 instances x tensor-parallel-size=2 on
#       the 4 GPUs, ports 8000-8001; CUDA graph enabled — no --enforce-eager)
#   [3] build + validate the controlled selections from
#       science_skill_annotations.json  -> vis-skill/skill_selections.json
#       (HARD GATE before any GPU work)
#   [4] text-skill   -> text-skill/
#   [5] cross-check: every text-skill episode used EXACTLY the annotation
#       top-5 (HARD GATE before the vis run)
#   [6] vis-skill    -> vis-skill/
#       (skill cards rendered by scripts/render_skill_cards.sh)
#
# Identical flow to ../qwen3.5-9b/run_all.sh — the
# ONLY difference is the backbone: models/Qwen3.5-27B instead of
# models/Qwen3.5-9B (served by 2 TP=2 instances instead of 4 TP=1
# instances; because the 27B model does not fit on one A100-80GB).
# Stage-1 skill retrieval is NOT done by the backbone at run time; both
# experiments use the SAME pre-annotated top-5 skills per episode
# (science_skill_annotations.json, model qwen3.8-max).
#
# One shared ONLINE vLLM deployment (2 servers, TP=2 each) is started once
# and reused by both experiments; each experiment's workers submit all
# episodes to ONE shared queue and round-robin every LLM request across all
# servers (true dynamic scheduling, high concurrency: a worker pulls the
# next episode the moment it finishes its current one, all GPUs stay busy
# until the last task).
#
# Hang-proof runners (inherited v1): hard per-episode wall-clock timeout +
# process-tree kill + retries; no unbounded future.result()/env.step()/exec
# waits.
#
# Summaries: summary.json already contains overall/per-task/per-category
# success rates — the compute_summary_with_categories.py logic is integrated
# into the runners' generate_summary().
#
# All runs are resumable: finished episodes (idx_*.json) are skipped, so
# re-running this script after an interruption continues where it stopped.
#
# Usage:
#   bash run_all.sh
#   nohup bash run_all.sh > run_all.nohup 2>&1 &   # recommended for long runs
# ============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"

BASE_DIR="$ROOT/baseline-noskill"
TEXT_DIR="$ROOT/text-skill"
VIS_DIR="$ROOT/vis-skill"

PYTHON="${PYTHON:-$(command -v python)}"
MODEL="${MODELS_DIR:-$REPO_ROOT/models}/Qwen3.5-27B"
MODEL_NAME="Qwen3.5-27B"
BASE_PORT=8000
NUM_PORTS=2

LOG="$ROOT/run_all.log"

echo "==========================================================================" | tee "$LOG"
echo " ScienceWorld 27B CONTROLLED experiments (qwen3.8-max annotation top-5):"  | tee -a "$LOG"
echo "   baseline-noskill -> text-skill -> vis-skill"                     | tee -a "$LOG"
echo " Started: $(date '+%F %T')"                                                | tee -a "$LOG"
echo "==========================================================================" | tee -a "$LOG"

# ── [0] Prerequisite checks ─────────────────────────────────────────────────
echo "[0/7] Checking prerequisites..." | tee -a "$LOG"
"$PYTHON" -c "from scienceworld import ScienceWorldEnv; print('  scienceworld OK')" 2>&1 | tee -a "$LOG"
"$PYTHON" -c "import vllm, openai, PIL, retry, yaml, psutil; print('  vllm/openai/PIL/retry/yaml/psutil OK')" 2>&1 | tee -a "$LOG"
java -version 2>&1 | head -1 | sed 's/^/  /' | tee -a "$LOG"
[ -d "$MODEL" ] || { echo "[ERROR] backbone not found: $MODEL" | tee -a "$LOG"; exit 1; }
[ -f "$TEXT_DIR/code/run_scienceworld_text.py" ] || { echo "[ERROR] missing text-skill scripts in $TEXT_DIR" | tee -a "$LOG"; exit 1; }
[ -f "$VIS_DIR/code/run_scienceworld_vis.py" ] || { echo "[ERROR] missing vis-skill scripts in $VIS_DIR" | tee -a "$LOG"; exit 1; }
[ -f "$TEXT_DIR/code/science_skill_annotations.json" ] || { echo "[ERROR] missing annotation in $TEXT_DIR/code/" | tee -a "$LOG"; exit 1; }
[ -f "$VIS_DIR/code/science_skill_annotations.json" ] || { echo "[ERROR] missing annotation in $VIS_DIR/code/" | tee -a "$LOG"; exit 1; }
[ -d "$VIS_DIR/images/skill_cards_v6" ] || { echo "[ERROR] missing skill cards in $VIS_DIR/images/ (run scripts/render_skill_cards.sh)" | tee -a "$LOG"; exit 1; }
echo "  prerequisites OK" | tee -a "$LOG"

# ── [1] Experiment 0: baseline-noskill (own vLLM server) ───────────────────
echo "[1/7] >>> baseline-noskill (no skill; TP=2 x DP=2 server on port $BASE_PORT)" | tee -a "$LOG"
bash "$BASE_DIR/launch.sh" 2>&1 | tee -a "$LOG"
bash "$BASE_DIR/stop_vllm.sh" 2>&1 | tee -a "$LOG"

# ── [2] Shared online vLLM servers (2 instances x TP=2, CUDA graph) ────────
SERVER_STARTED=0
ALL_HEALTHY=true
for i in $(seq 0 $((NUM_PORTS - 1))); do
    if ! curl -s "http://localhost:$((BASE_PORT + i))/health" > /dev/null 2>&1; then
        ALL_HEALTHY=false; break
    fi
done
if [ "$ALL_HEALTHY" = true ]; then
    # Healthy — but they MUST serve the 27B backbone. Ports occupied by a
    # different model (e.g. the 9B experiment) must not be silently reused.
    WRONG_MODEL=false
    for i in $(seq 0 $((NUM_PORTS - 1))); do
        PORT=$((BASE_PORT + i))
        if ! curl -s "http://localhost:$PORT/v1/models" | grep -q "$MODEL"; then
            WRONG_MODEL=true
            echo "[ERROR] port $PORT is occupied by a DIFFERENT model:" | tee -a "$LOG"
            curl -s "http://localhost:$PORT/v1/models" | head -c 300 | tee -a "$LOG" | sed 's/^/ /'
            echo "" | tee -a "$LOG"
        fi
    done
    if [ "$WRONG_MODEL" = true ]; then
        echo "[ERROR] Free ports ${BASE_PORT}-$((BASE_PORT + NUM_PORTS - 1)) (stop the other vLLM servers) before starting the controlled 27B experiments." | tee -a "$LOG"
        exit 1
    fi
    echo "[2/7] Reusing existing healthy vLLM servers (serving $MODEL) on ports ${BASE_PORT}-$((BASE_PORT + NUM_PORTS - 1))." | tee -a "$LOG"
else
    echo "[2/7] Starting $NUM_PORTS online vLLM servers (TP=2 each, ports ${BASE_PORT}-$((BASE_PORT + NUM_PORTS - 1)))..." | tee -a "$LOG"
    bash "$ROOT/start_vllm.sh" 2>&1 | tee -a "$LOG"
    SERVER_STARTED=1
fi

cleanup() {
    echo "[run_all] Cleanup (servers started by this run: $SERVER_STARTED)..." | tee -a "$LOG" || true
    if [ "$SERVER_STARTED" = "1" ]; then
        bash "$ROOT/stop_vllm.sh" 2>&1 | tee -a "$LOG" || true
    fi
}
trap cleanup EXIT

# ── [3] Build + validate the controlled selections (hard gate) ─────────────
echo "[3/7] >>> building controlled skill selections from science_skill_annotations.json (top-5)" | tee -a "$LOG"
"$PYTHON" "$VIS_DIR/code/prepare_skill_selections.py" \
    --annotations "$VIS_DIR/code/science_skill_annotations.json" \
    --indices "$VIS_DIR/code/test_indices.json" \
    --skills-dir "$VIS_DIR/skills_scienceworld" \
    --out "$VIS_DIR/skill_selections.json" 2>&1 | tee -a "$LOG"

# ── [4] Experiment 1: text-skill (controlled) ──────────────────────────────
echo "[4/7] >>> text-skill controlled (SkillNet pipeline, SKILL.md text, annotation top-5)" | tee -a "$LOG"
bash "$TEXT_DIR/launch.sh" 2>&1 | tee -a "$LOG"

# ── [5] Cross-check text-skill vs annotation selections (hard gate) ─────────
echo "[5/7] >>> cross-checking text-skill selections against the annotation top-5" | tee -a "$LOG"
"$PYTHON" "$VIS_DIR/code/prepare_skill_selections.py" \
    --annotations "$VIS_DIR/code/science_skill_annotations.json" \
    --indices "$VIS_DIR/code/test_indices.json" \
    --skills-dir "$VIS_DIR/skills_scienceworld" \
    --text-results "$TEXT_DIR/results/scienceworld/$MODEL_NAME/text_skill" \
    --out "$VIS_DIR/skill_selections.json" 2>&1 | tee -a "$LOG"

# ── [6] Experiment 2: vis-skill (controlled) ────────────────────────────────
echo "[6/7] >>> vis-skill controlled (same per-episode skills as text-skill, rendered as images)" | tee -a "$LOG"
bash "$VIS_DIR/launch.sh" 2>&1 | tee -a "$LOG"

# ── [7] Final summaries ─────────────────────────────────────────────────────
# summary.json already contains success_count/success_rate and by_category
# (compute_summary_with_categories.py logic integrated into generate_summary).
echo "" | tee -a "$LOG"
echo "[7/7] ========================= FINAL SUMMARIES =========================" | tee -a "$LOG"
for d in "$BASE_DIR/results/scienceworld/$MODEL_NAME/noskill_baseline" \
         "$TEXT_DIR/results/scienceworld/$MODEL_NAME/text_skill" \
         "$VIS_DIR/results/scienceworld/$MODEL_NAME/vis_skill"; do
    if [ -f "$d/summary.json" ]; then
        echo "── $d ──" | tee -a "$LOG"
        "$PYTHON" - "$d/summary.json" <<'EOF' | tee -a "$LOG"
import json, sys
s = json.load(open(sys.argv[1]))
o = s["overall"]
print(f"  {s['experiment']}")
print(f"  total={o['total']}  avg_score={o['avg_score']}  "
      f"avg_steps={o['avg_steps']}  task_done={o['task_done']} ({o['task_done_rate']}%)  "
      f"success={o['success_count']} ({o['success_rate']}%)")
bc = s.get("by_category", {})
if bc:
    print("  by_category:")
    for cat, c in bc.items():
        print(f"    {cat:<8} n={c['count']:>3}  score={c['avg_score']:>6.2f}  "
              f"succ={c['success_count']}/{c['count']} ({c['success_rate']}%)  "
              f"steps={c['avg_steps']:>5.1f}")
EOF
    else
        echo "── [WARN] missing summary: $d/summary.json ──" | tee -a "$LOG"
    fi
done
echo "==========================================================================" | tee -a "$LOG"
echo "All three experiments finished: $(date '+%F %T')" | tee -a "$LOG"
