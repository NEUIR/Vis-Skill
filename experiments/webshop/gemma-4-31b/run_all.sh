#!/bin/bash
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
# =============================================================================
# run_all.sh — Master script: run ALL THREE WebShop experiments for
# Gemma-4-31B, one-shot, in the required order:
#
#   1. baseline-noskill   (no skill, direct task execution)
#   2. text-skill         (CONTROLLED: pre-annotated top-5 skills per task,
#                          SKILL.md text only — no model-based skill selection)
#   3. CONFIRM the annotation-based skill selections that vis-skill will use
#      (top-5 of webshop_skill_annotations.json must cover all 200
#      games; distribution printed)
#   4. vis-skill          (rendered skill card images; uses the SAME top-5
#                          annotation selections -> every task uses exactly
#                          the same skills as in the text-skill run)
#
# The WebShop dataset lives ONCE in the shared dir experiments/webshop/data
# (a symlink to data/webshop/data created by scripts/setup_data.sh); all
# experiments read it from there (no per-experiment copy).
#
# Deployment: ONLINE vLLM, TP=2 (Gemma-4-31B is ~62.5GB of bf16 weights,
# too big for a single A100-80GB at 65K context), so 4 GPUs = 2 instances:
#   instance 0: GPUs 0,1 -> port 8000
#   instance 1: GPUs 2,3 -> port 8001
#
# Each experiment:
#   a) starts its own ONLINE vLLM servers (2 instances x TP=2,
#      gpu-memory-utilization 0.90, max-model-len 65536)
#   b) runs the evaluation with TRUE dynamic scheduling
#      (one shared queue; a worker pulls the next task as soon as it
#       finishes its current one, so all GPUs stay busy until the end)
#   c) stops ITS OWN vLLM servers before the next experiment starts
#
# Safety:
#   - only kills vLLM servers that belong to THIS Gemma-4-31B experiment set
#   - refuses to start if GPUs are still occupied by another job
#   - supports resume: re-running skips games whose idx_*.json already exists
#
# Usage:
#   nohup bash run_all.sh > run_all.log 2>&1 &
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_PREFIX="${ENV_PREFIX:-${CONDA_PREFIX:-$REPO_ROOT/envs/vis-skill-webshop-gemma}}"   # active conda env, else the create_env.sh prefix
PYTHON="$ENV_PREFIX/bin/python"
SHARED_DATA="$SCRIPT_DIR/../data"   # experiments/webshop/data (shared dataset dir)
ANNOTATIONS="$REPO_ROOT/annotation/outputs/webshop_skill_annotations.json"
MODEL_DIR="${MODELS_DIR:-$REPO_ROOT/models}/Gemma-4-31B-it"

echo "============================================================"
echo " Gemma-4-31B WebShop — Full Experiment Pipeline"
echo " Started at: $(date)"
echo "============================================================"

# ─────────────────────────────────────────────────────────────────
# Pre-flight checks
# ─────────────────────────────────────────────────────────────────
if [ ! -x "$PYTHON" ]; then
    echo "[ERROR] $PYTHON not found. Run first: bash create_env.sh"
    exit 1
fi
for f in items_shuffle.json items_ins_v2.json items_human_ins.json \
         webshop_overall_procedure_examples.txt webshop_procedure_code_template.py; do
    if [ ! -f "$SHARED_DATA/$f" ]; then
        echo "[ERROR] $SHARED_DATA/$f missing (unified dataset dir)."
        exit 1
    fi
done
if [ ! -f "$ANNOTATIONS" ]; then
    echo "[ERROR] $ANNOTATIONS missing (pre-annotated skill selections)."
    exit 1
fi

# Model weights must be COMPLETELY downloaded before serving:
# every shard listed in model.safetensors.index.json must exist.
MISSING_SHARDS=$("$PYTHON" - "$MODEL_DIR" <<'PYEOF'
import json, os, sys
model_dir = sys.argv[1]
index_path = os.path.join(model_dir, "model.safetensors.index.json")
if not os.path.exists(index_path):
    print("model.safetensors.index.json")
    sys.exit(0)
with open(index_path) as f:
    index = json.load(f)
shards = sorted(set(index["weight_map"].values()))
missing = [s for s in shards if not os.path.exists(os.path.join(model_dir, s))]
print(" ".join(missing))
PYEOF
)
if [ -n "$MISSING_SHARDS" ]; then
    echo "[ERROR] $MODEL_DIR download incomplete — missing shard(s): $MISSING_SHARDS"
    echo "        Wait for the model download to finish, then re-run this script."
    exit 1
fi

# GPUs must be free (this pipeline needs all 4 A100s; ~192GB RAM for workers)
BUSY=$("$PYTHON" - <<'PYEOF'
import subprocess
out = subprocess.run(
    ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
    capture_output=True, text=True).stdout
busy = [l.split(",")[0].strip() for l in out.strip().splitlines()
        if float(l.split(",")[1]) > 5000]
print(" ".join(busy))
PYEOF
)
if [ -n "$BUSY" ]; then
    echo "[ERROR] GPU(s) $BUSY still have >5GB in use by another job."
    echo "        Wait for it to finish (or stop it), then re-run this script."
    exit 1
fi
echo "[preflight] env + data + model shards present, all 4 GPUs free."

# ─────────────────────────────────────────────────────────────────
# Helper: kill any LEFTOVER Gemma-4-31B-it vLLM servers of this experiment set
# (never touches other experiments' servers)
# ─────────────────────────────────────────────────────────────────
kill_gemma_vllm() {
    local PIDS
    PIDS=$(pgrep -f "vllm.entrypoints.openai.api_server.*Gemma-4-31B-it" || true)
    if [ -n "$PIDS" ]; then
        echo "[cleanup] stopping leftover Gemma-4-31B-it vLLM servers: $PIDS"
        for PID in $PIDS; do kill "$PID" 2>/dev/null || true; done
        sleep 15
    fi
}

# ─────────────────────────────────────────────────────────────────
# EXPERIMENT 1: baseline-noskill
# ─────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " [1/4] BASELINE-NOSKILL            $(date)"
echo "============================================================"
kill_gemma_vllm
bash "$SCRIPT_DIR/baseline-noskill/start_vllm.sh"
bash "$SCRIPT_DIR/baseline-noskill/launch.sh"
bash "$SCRIPT_DIR/baseline-noskill/stop_vllm.sh"
echo "[1/4] baseline-noskill COMPLETE at $(date)"

# ─────────────────────────────────────────────────────────────────
# EXPERIMENT 2: text-skill
# ─────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " [2/4] TEXT-SKILL (CONTROLLED: annotated top-5, SKILL.md only)   $(date)"
echo "============================================================"
kill_gemma_vllm
bash "$SCRIPT_DIR/text-skill/start_vllm.sh"
bash "$SCRIPT_DIR/text-skill/launch.sh"
bash "$SCRIPT_DIR/text-skill/stop_vllm.sh"
echo "[2/4] text-skill COMPLETE at $(date)"

# ─────────────────────────────────────────────────────────────────
# GATE: confirm the annotation-based skill selections used by vis-skill
# (same source + top-K rule as text-skill, so both match by construction)
# ─────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " [3/4] CONFIRM ANNOTATION-BASED SKILL SELECTIONS (top-5)"
echo "============================================================"
"$PYTHON" - "$ANNOTATIONS" <<'PYEOF'
import collections, json, sys
with open(sys.argv[1]) as f:
    data = json.load(f)
TOP_K = 5
by_index = {item["index"]: item for item in data}
missing = [i for i in range(200) if i not in by_index]
assert not missing, f"annotations missing for games: {missing}"
sel = {}
for i in range(200):
    ranking = by_index[i]["annotation"]["skill_ranking"]
    sel[i] = [s["skill_name"] for s in sorted(ranking, key=lambda x: x["rank"])[:TOP_K]]
dist = collections.Counter(len(v) for v in sel.values())
names = collections.Counter(n for v in sel.values() for n in v)
print(f"[confirm] all 200 games have a top-{TOP_K} skill selection in the annotations")
print(f"[confirm] skills-per-task distribution: {dict(sorted(dist.items()))}")
print(f"[confirm] per-skill usage counts:")
for name, cnt in sorted(names.items()):
    print(f"          {name:<40} {cnt}")
PYEOF
echo "[3/4] selections confirmed from: $ANNOTATIONS"

# ─────────────────────────────────────────────────────────────────
# EXPERIMENT 3: vis-skill (uses the same annotation selections)
# ─────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " [4/4] VIS-SKILL (rendered skill cards, multimodal)   $(date)"
echo "============================================================"
kill_gemma_vllm
bash "$SCRIPT_DIR/vis-skill/start_vllm.sh"
bash "$SCRIPT_DIR/vis-skill/launch.sh"
bash "$SCRIPT_DIR/vis-skill/stop_vllm.sh"
echo "[4/4] vis-skill COMPLETE at $(date)"

# ─────────────────────────────────────────────────────────────────
# DONE
# ─────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " ALL EXPERIMENTS COMPLETE — $(date)"
echo "============================================================"
R1="$SCRIPT_DIR/baseline-noskill/results/Gemma-4-31B-it/gemma_webshop_baseline_skill_False/summary.json"
R2="$SCRIPT_DIR/text-skill/results/Gemma-4-31B-it/gemma_webshop_textskill_skill_True/summary.json"
R3="$SCRIPT_DIR/vis-skill/results/Gemma-4-31B-it/gemma_webshop_visskill_skill_True/summary.json"
for R in "$R1" "$R2" "$R3"; do
    if [ -f "$R" ]; then
        echo ""; echo "== $R =="; cat "$R"
    fi
done
echo ""
echo "Per-game results + logs live in each experiment's results/ dir and logs_run.log."
