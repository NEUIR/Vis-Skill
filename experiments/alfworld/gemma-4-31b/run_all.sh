#!/bin/bash
# =============================================================================
# Master script: run ALL THREE experiments for Gemma-4-31B on ALFWorld.
#
# Order:
#   1. baseline-noskill  (no skill, direct task execution)
#   2. text-skill        (SkillNet flow, SKILL.md text only)
#   3. vis-skill         (rendered skill card images, same top-5 annotation)
#
# Each experiment:
#   a) starts its own ONLINE vLLM servers (2 instances x TP=2, gpu-util=0.9,
#      max-model-len=24576; 31B bf16 weights ~62.5GB need a GPU pair each)
#   b) runs the evaluation with TRUE dynamic scheduling (one shared queue,
#      high concurrency, round-robin port assignment at task pick-up)
#   c) stops the vLLM servers before moving to the next experiment
#
# The vis-skill experiment uses the same annotation-based top-5 skill
# selection as text-skill (both read code/alfworld_skill_ranking_per_instance.json),
# so the three experiments are fully comparable.
#
# SAFETY: this script only ever kills vLLM servers serving Gemma-4-31B and
# refuses to start while ANY other process occupies the GPUs.
#
# Usage:
#   nohup bash run_all.sh > run_all.log 2>&1 &
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$SCRIPT_DIR"

echo "============================================================"
echo " Gemma-4-31B ALFWorld — Full Experiment Pipeline"
echo " Started at: $(date)"
echo "============================================================"
echo ""

# ─────────────────────────────────────────────────────────────────
# Pre-flight: ALL 4 GPUs must be free (this pipeline needs all of
# them and must not disturb any other experiment).
# ─────────────────────────────────────────────────────────────────
BUSY=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
if [ "$BUSY" -gt 0 ]; then
    echo "[ERROR] $BUSY process(es) currently occupy the GPUs."
    echo "        This pipeline needs all 4 GPUs exclusively. Wait for the"
    echo "        other experiment to finish (or stop it) and re-run:"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null || true
    exit 1
fi

# ─────────────────────────────────────────────────────────────────
# Helper: kill leftover Gemma-4-31B vLLM servers on our ports
# (never touches servers of other models/experiments)
# ─────────────────────────────────────────────────────────────────
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

kill_all_vllm() {
    echo "[cleanup] Stopping any leftover Gemma-4-31B vLLM servers on ports 8000-8001..."
    for PORT in 8000 8001; do
        PIDS=$(pgrep -f "vllm.entrypoints.openai.api_server.*Gemma-4-31B.*--port $PORT" || true)
        for PID in $PIDS; do
            echo "  Killing PID $PID (port $PORT)"
            kill_tree "$PID"
        done
    done
    sleep 10
    echo "[cleanup] Done."
}

# ─────────────────────────────────────────────────────────────────
# EXPERIMENT 1: baseline-noskill
# ─────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " [1/3] BASELINE-NOSKILL"
echo " $(date)"
echo "============================================================"

kill_all_vllm

echo "[1/3] Starting vLLM servers for baseline-noskill..."
bash "$SCRIPT_DIR/baseline-noskill/start_vllm.sh"

echo "[1/3] Running baseline-noskill evaluation..."
bash "$SCRIPT_DIR/baseline-noskill/launch.sh"

echo "[1/3] Stopping vLLM servers..."
bash "$SCRIPT_DIR/baseline-noskill/stop_vllm.sh"

echo "[1/3] baseline-noskill COMPLETE at $(date)"

# ─────────────────────────────────────────────────────────────────
# EXPERIMENT 2: text-skill
# ─────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " [2/3] TEXT-SKILL"
echo " $(date)"
echo "============================================================"

kill_all_vllm

echo "[2/3] Starting vLLM servers for text-skill..."
bash "$SCRIPT_DIR/text-skill/start_vllm.sh"

echo "[2/3] Running text-skill evaluation..."
bash "$SCRIPT_DIR/text-skill/launch.sh"

echo "[2/3] Stopping vLLM servers..."
bash "$SCRIPT_DIR/text-skill/stop_vllm.sh"

echo "[2/3] text-skill COMPLETE at $(date)"

# ─────────────────────────────────────────────────────────────────
# EXPERIMENT 3: vis-skill
# ─────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " [3/3] VIS-SKILL"
echo " $(date)"
echo "============================================================"

kill_all_vllm

echo "[3/3] Starting vLLM servers for vis-skill (multimodal)..."
bash "$SCRIPT_DIR/vis-skill/start_vllm.sh"

echo "[3/3] Running vis-skill evaluation..."
bash "$SCRIPT_DIR/vis-skill/launch.sh"

echo "[3/3] Stopping vLLM servers..."
bash "$SCRIPT_DIR/vis-skill/stop_vllm.sh"

echo "[3/3] vis-skill COMPLETE at $(date)"

# ─────────────────────────────────────────────────────────────────
# DONE
# ─────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " ALL EXPERIMENTS COMPLETE"
echo " Finished at: $(date)"
echo "============================================================"
echo ""
echo "Results:"
echo "  baseline-noskill: $SCRIPT_DIR/baseline-noskill/results/alfworld/Gemma-4-31B-it/noskill_baseline/"
echo "  text-skill:       $SCRIPT_DIR/text-skill/results/alfworld/Gemma-4-31B-it/text_baseline/"
echo "  vis-skill:        $SCRIPT_DIR/vis-skill/results/alfworld/Gemma-4-31B-it/vis_baseline/"
