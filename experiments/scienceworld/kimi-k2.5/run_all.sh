#!/bin/bash
# ============================================================================
# ONE-SHOT launcher for the ScienceWorld x kimi-k2.5 experiments via the
# DashScope API (NO local vLLM — all inference goes to the API).
#
# Skill selections are fixed by the qwen3.8-max pre-annotation (top-5
# relevance per episode, model-independent):
#
#   [0] prerequisite checks (python env, java, DASHSCOPE_API_KEY)
#   [1] shared datasets via shared/ (symlinks, no duplication)
#   [2] build + validate the controlled selections from
#       science_skill_annotations.json -> shared/skill_selections.json
#       (HARD GATE before any API work)
#   [3] baseline-noskill  -> baseline-noskill/
#   [4] text-skill        -> text-skill/
#   [5] cross-check: every text-skill episode used EXACTLY the annotation
#       top-5 (HARD GATE before the vis run)
#   [6] vis-skill         -> vis-skill/ (pre-rendered cards, shared)
#   [7] final summaries
#
# Every experiment's workers submit all 211 episodes to ONE shared queue
# (true dynamic scheduling, 20 concurrent workers), and all runs are
# resumable: finished episodes (idx_*.json) are skipped. Hang-proof runners:
# hard per-episode wall-clock timeout + process-tree kill + retries.
#
# Usage:
#   export DASHSCOPE_API_KEY=<your-key>
#   nohup bash run_all.sh > run_all.nohup 2>&1 &   # recommended for long runs
# ============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
SHARED_DIR="$ROOT/shared"

PYTHON="${PYTHON:-$(command -v python)}"
MODEL="kimi-k2.5"
MODEL_NAME="kimi-k2.5"

LOG="$ROOT/run_all.log"

echo "==========================================================================" | tee "$LOG"
echo " ScienceWorld x $MODEL experiments (DashScope API,"           | tee -a "$LOG"
echo " qwen3.8-max annotation top-5 fixed selections):"                      | tee -a "$LOG"
echo "   baseline-noskill -> text-skill -> vis-skill (same fixed selections)" | tee -a "$LOG"
echo " Started: $(date '+%F %T')"                                              | tee -a "$LOG"
echo "==========================================================================" | tee -a "$LOG"

# ── [0] Prerequisite checks ─────────────────────────────────────────────────
echo "[0/7] Checking prerequisites..." | tee -a "$LOG"
[ -x "$PYTHON" ] || { echo "[ERROR] $PYTHON not found (vis-skill-science env required)" | tee -a "$LOG"; exit 1; }
"$PYTHON" -c "from scienceworld import ScienceWorldEnv; print('  scienceworld OK')" 2>&1 | tee -a "$LOG"
"$PYTHON" -c "import openai, retry, yaml, psutil, tqdm; print('  openai/retry/yaml/psutil/tqdm OK')" 2>&1 | tee -a "$LOG"
java -version 2>&1 | head -1 | sed 's/^/  /' | tee -a "$LOG"
if [ -z "${DASHSCOPE_API_KEY:-}" ]; then
    echo "[ERROR] DASHSCOPE_API_KEY is not set." | tee -a "$LOG"
    echo "        export DASHSCOPE_API_KEY=<your-key>" | tee -a "$LOG"
    exit 1
fi
BASE_URL="${DASHSCOPE_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
    -H "Authorization: Bearer $DASHSCOPE_API_KEY" \
    "$BASE_URL/models")
if [ "$HTTP_CODE" != "200" ]; then
    echo "[WARN] DashScope API check returned HTTP $HTTP_CODE ($BASE_URL/models)." | tee -a "$LOG"
    echo "       Continuing anyway; if the key/network is bad the run will fail." | tee -a "$LOG"
else
    echo "  DashScope API reachable: $BASE_URL (HTTP $HTTP_CODE)" | tee -a "$LOG"
fi
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

# ── [3] Experiment 1: baseline-noskill ─────────────────────────────────────
echo "[3/7] >>> baseline-noskill (no skill, direct Thought/Action execution)" | tee -a "$LOG"
bash "$ROOT/baseline-noskill/launch.sh" 2>&1 | tee -a "$LOG"

# ── [4] Experiment 2: text-skill (controlled) ──────────────────────────────
echo "[4/7] >>> text-skill controlled (SkillNet pipeline, SKILL.md text, annotation top-5)" | tee -a "$LOG"
bash "$ROOT/text-skill/launch.sh" 2>&1 | tee -a "$LOG"

# ── [5] Cross-check text-skill vs annotation selections (hard gate) ─────────
echo "[5/7] >>> cross-checking text-skill selections against the annotation top-5" | tee -a "$LOG"
"$PYTHON" "$SHARED_DIR/code/prepare_skill_selections.py" \
    --annotations "$SHARED_DIR/science_skill_annotations.json" \
    --indices "$SHARED_DIR/test_indices.json" \
    --skills-dir "$SHARED_DIR/skills_scienceworld" \
    --text-results "$ROOT/text-skill/results/scienceworld/$MODEL_NAME/text_skill" \
    --out "$SHARED_DIR/skill_selections.json" 2>&1 | tee -a "$LOG"

# ── [6] Experiment 3: vis-skill (controlled) ────────────────────────────────
echo "[6/7] >>> vis-skill controlled (same per-episode skills as text-skill, rendered as images)" | tee -a "$LOG"
bash "$ROOT/vis-skill/launch.sh" 2>&1 | tee -a "$LOG"

# ── [7] Final summaries ─────────────────────────────────────────────────────
echo "" | tee -a "$LOG"
echo "======================= FINAL SUMMARIES ($MODEL) =======================" | tee -a "$LOG"
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
echo "All three $MODEL experiments finished: $(date '+%F %T')" | tee -a "$LOG"
