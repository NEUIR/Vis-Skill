#!/bin/bash
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
# ============================================================================
# ONE-SHOT launcher for the ScienceWorld Gemma-4-31B experiments
# (skill selection fixed by the qwen3.8-max pre-annotation, top-5 relevance;
# the selection is model-independent and lives in shared/):
#
#   [1] shared datasets via shared/ (symlinks, no duplication)
#   [2] build + validate the controlled selections from
#       science_skill_annotations.json -> shared/skill_selections.json
#       (HARD GATE before any GPU work)
#   [3] shared ONLINE vLLM servers (2 GPU pairs, TP=2, ports 8000-8001,
#       CUDA graph acceleration ON, multimodal flags) — ONE deployment
#       reused by ALL THREE experiments
#   [4] baseline-noskill  -> baseline-noskill
#   [5] text-skill        -> text-skill
#   [6] cross-check: every text-skill episode used EXACTLY the annotation
#       top-5 (HARD GATE before the vis run)
#   [7] vis-skill         -> vis-skill (cards rendered by scripts/render_skill_cards.sh)
#
# One shared ONLINE vLLM deployment is started once and reused by all three
# experiments; each experiment's workers submit all 211 episodes to ONE
# shared queue and round-robin every LLM request across both servers
# (true dynamic scheduling at episode level AND request level, high
# concurrency: a worker pulls the next episode the moment it finishes its
# current one, all GPUs stay busy until the last task).
#
# SAFETY: refuses to start while ANY other process occupies the GPUs;
# only leftover Gemma-4-31B servers on our ports are killed.
#
# Hang-proof runners: hard per-episode wall-clock timeout + process-tree
# kill + retries; no unbounded waits. All runs are resumable: finished
# episodes (idx_*.json) are skipped.
#
# Usage:
#   bash run_all.sh
#   nohup bash run_all.sh > run_all.nohup 2>&1 &   # recommended for long runs
# ============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
SHARED_DIR="$ROOT/shared"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

echo "=================================================================="
echo "[$(timestamp)] === ScienceWorld x Gemma — 31B EXPERIMENT BATCH START ==="
echo "=================================================================="

# ── Pre-flight: ALL GPUs must be free (this batch needs all 4 exclusively) ──
BUSY=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
if [ "$BUSY" -gt 0 ]; then
    echo "[ERROR] $BUSY process(es) currently occupy the GPUs."
    echo "        This batch needs all GPUs exclusively. Wait for the other"
    echo "        experiment to finish (or stop it) and re-run."
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null || true
    exit 1
fi

kill_tree() {
    local pid=$1
    local children
    children=$(pgrep -P "$pid" 2>/dev/null || true)
    local c
    for c in $children; do
        kill_tree "$c"
    done
    kill -9 "$pid" 2>/dev/null || true
}

# Kill only leftover vLLM servers serving a SPECIFIC Gemma backbone on our
# ports (never touches other experiments' servers).
kill_gemma_vllm() {
    local MODEL_PATTERN="$1"
    echo "[$(timestamp)] Stopping any leftover $MODEL_PATTERN vLLM servers on ports 8000-8003..."
    for PORT in 8000 8001 8002 8003; do
        PIDS=$(pgrep -f "vllm.entrypoints.openai.api_server.*$MODEL_PATTERN.*--port $PORT" || true)
        for PID in $PIDS; do
            echo "  Killing PID $PID (port $PORT)"
            kill_tree "$PID"
        done
    done
    sleep 20
    for PORT in 8000 8001 8002 8003; do
        if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
            echo "[$(timestamp)] [ERROR] port $PORT still alive after cleanup!"
            exit 1
        fi
    done
    echo "[$(timestamp)] Cleanup done — all ports clear."
}

# ── Pre-flight: ALL GPUs must be free (this batch needs all 4 exclusively) ──
BUSY=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
if [ "$BUSY" -gt 0 ]; then
    echo "[ERROR] $BUSY process(es) currently occupy the GPUs."
    echo "        This batch needs all GPUs exclusively. Wait for the other"
    echo "        experiment to finish (or stop it) and re-run."
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null || true
    exit 1
fi

kill_gemma_vllm "Gemma-4-31B-it"


PYTHON="${PYTHON:-$(command -v python)}"
MODEL="${MODELS_DIR:-$REPO_ROOT/models}/Gemma-4-31B-it"
MODEL_NAME="Gemma-4-31B-it"
BASE_PORT=8000
NUM_PORTS=2

LOG="$ROOT/run_all.log"

echo "==========================================================================" | tee "$LOG"
echo " ScienceWorld Gemma-4-31B experiments (qwen3.8-max annotation top-5):"     | tee -a "$LOG"
echo "   baseline-noskill -> text-skill -> vis-skill (same fixed selections)"   | tee -a "$LOG"
echo " Started: $(date '+%F %T')"                                                | tee -a "$LOG"
echo "==========================================================================" | tee -a "$LOG"

# ── [0] Prerequisite checks ─────────────────────────────────────────────────
echo "[0/7] Checking prerequisites..." | tee -a "$LOG"
[ -x "$PYTHON" ] || { echo "[ERROR] $PYTHON not found — run: bash $ROOT/create_env.sh" | tee -a "$LOG"; exit 1; }
"$PYTHON" -c "from scienceworld import ScienceWorldEnv; print('  scienceworld OK')" 2>&1 | tee -a "$LOG"
"$PYTHON" -c "import vllm, openai, PIL, retry, yaml, psutil, httpx; print('  vllm/openai/PIL/retry/yaml/psutil/httpx OK')" 2>&1 | tee -a "$LOG"
java -version 2>&1 | head -1 | sed 's/^/  /' | tee -a "$LOG"
[ -d "$MODEL" ] || { echo "[ERROR] backbone not found: $MODEL" | tee -a "$LOG"; exit 1; }
echo "  prerequisites OK" | tee -a "$LOG"

# ── [1] Shared datasets (symlinks; idempotent) ─────────────────────────────
echo "[1/7] Setting up shared datasets (symlinks, no duplication)..." | tee -a "$LOG"
bash "$SHARED_DIR/setup_shared.sh" 2>&1 | tee -a "$LOG"

# ── [2] Build + validate the controlled selections (hard gate) ─────────────
echo "[2/7] >>> building controlled skill selections from science_skill_annotations.json (top-5)" | tee -a "$LOG"
"$PYTHON" "$SHARED_DIR/code/prepare_skill_selections.py" \
    --annotations "$SHARED_DIR/science_skill_annotations.json" \
    --indices "$SHARED_DIR/test_indices.json" \
    --skills-dir "$SHARED_DIR/skills_scienceworld" \
    --out "$SHARED_DIR/skill_selections.json" 2>&1 | tee -a "$LOG"

# ── [3] Shared online vLLM servers (one per GPU, CUDA graphs ON) ───────────
SERVER_STARTED=0
ALL_HEALTHY=true
for i in $(seq 0 $((NUM_PORTS - 1))); do
    if ! curl -s "http://localhost:$((BASE_PORT + i))/health" > /dev/null 2>&1; then
        ALL_HEALTHY=false; break
    fi
done
if [ "$ALL_HEALTHY" = true ]; then
    # Healthy — but they MUST serve the 31B backbone (start_vllm.sh itself
    # also enforces this; double-check here for the log).
    for i in $(seq 0 $((NUM_PORTS - 1))); do
        PORT=$((BASE_PORT + i))
        if ! curl -s "http://localhost:$PORT/v1/models" | grep -q "$MODEL"; then
            echo "[ERROR] port $PORT is occupied by a DIFFERENT model:" | tee -a "$LOG"
            curl -s "http://localhost:$PORT/v1/models" | head -c 300 | tee -a "$LOG" | sed 's/^/ /'
            echo "" | tee -a "$LOG"
            echo "[ERROR] Free ports ${BASE_PORT}-$((BASE_PORT + NUM_PORTS - 1)) first." | tee -a "$LOG"
            exit 1
        fi
    done
    echo "[3/7] Reusing existing healthy vLLM servers (serving $MODEL) on ports ${BASE_PORT}-$((BASE_PORT + NUM_PORTS - 1))." | tee -a "$LOG"
else
    echo "[3/7] Starting $NUM_PORTS online vLLM servers (one per GPU pair (TP=2), ports ${BASE_PORT}-$((BASE_PORT + NUM_PORTS - 1)), CUDA graphs ON)..." | tee -a "$LOG"
    bash "$ROOT/start_vllm.sh" 2>&1 | tee -a "$LOG"
    SERVER_STARTED=1
fi

cleanup() {
    echo "[run_all] Cleanup (servers started by this run: $SERVER_STARTED)..." | tee -a "$LOG" || true
    if [ "$SERVER_STARTED" = "1" ]; then
        bash "$ROOT/stop_vllm.sh" 2>&1 | tee -a "$LOG" || true
    fi
    kill_gemma_vllm "Gemma-4-31B-it"
}
trap cleanup EXIT

# ── [4] Experiment 1: baseline-noskill ─────────────────────────────────────
echo "[4/7] >>> baseline-noskill (no skill, direct Thought/Action execution)" | tee -a "$LOG"
bash "$ROOT/baseline-noskill/launch.sh" 2>&1 | tee -a "$LOG"

# ── [5] Experiment 2: text-skill (controlled) ──────────────────────────────
echo "[5/7] >>> text-skill controlled (SkillNet pipeline, SKILL.md text, annotation top-5)" | tee -a "$LOG"
bash "$ROOT/text-skill/launch.sh" 2>&1 | tee -a "$LOG"

# ── [6] Cross-check text-skill vs annotation selections (hard gate) ─────────
echo "[6/7] >>> cross-checking text-skill selections against the annotation top-5" | tee -a "$LOG"
"$PYTHON" "$SHARED_DIR/code/prepare_skill_selections.py" \
    --annotations "$SHARED_DIR/science_skill_annotations.json" \
    --indices "$SHARED_DIR/test_indices.json" \
    --skills-dir "$SHARED_DIR/skills_scienceworld" \
    --text-results "$ROOT/text-skill/results/scienceworld/$MODEL_NAME/text_skill" \
    --out "$SHARED_DIR/skill_selections.json" 2>&1 | tee -a "$LOG"

# ── [7] Experiment 3: vis-skill (controlled) ────────────────────────────────
echo "[7/7] >>> vis-skill controlled (same per-episode skills as text-skill, rendered as images)" | tee -a "$LOG"
bash "$ROOT/vis-skill/launch.sh" 2>&1 | tee -a "$LOG"

# ── Final summaries ─────────────────────────────────────────────────────────
echo "" | tee -a "$LOG"
echo "========================= FINAL SUMMARIES (Gemma-4-31B) =========================" | tee -a "$LOG"
for d in "$ROOT/baseline-noskill/results/scienceworld/$MODEL_NAME/noskill_baseline" \
         "$ROOT/text-skill/results/scienceworld/$MODEL_NAME/text_skill" \
         "$ROOT/vis-skill/results/scienceworld/$MODEL_NAME/vis_skill"; do
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
echo "All three Gemma-4-31B experiments finished: $(date '+%F %T')" | tee -a "$LOG"
