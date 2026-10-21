#!/bin/bash
# =============================================================================
# Master script: run the three Qwen3.5-27B experiments on ALFWorld.
#
# Order (independent — both skill runs use the qwen3.8-max annotation for
# skill selection, no cross-dependency):
#   1. baseline-noskill (no skill, direct task execution)
#   2. text-skill       (SkillNet flow, SKILL.md text only)
#   3. vis-skill        (SkillNet flow, v5 rendered skill card images)
#
# Each experiment:
#   a) starts its own ONLINE vLLM servers (2 replicas x TP=2, all 4 GPUs)
#   b) runs the evaluation with TRUE dynamic scheduling
#      (32 workers for baseline-noskill, 128 for the skill runs, pulling from
#      ONE shared queue, round-robin over ports)
#   c) stops the vLLM servers before moving to the next experiment
#
# The script first WAITS until every GPU is idle before deploying.
#
# Usage:
#   nohup bash run_all.sh > run_all.log 2>&1 &
# =============================================================================
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo " Qwen3.5-27B ALFWorld — baseline-noskill + text-skill + vis-skill"
echo " Started at: $(date)"
echo "============================================================"

# ── Wait until all 4 GPUs are idle (memory.used < 2000 MiB) and the ports
#    used by the previous experiment stop serving ──────────────────────────
wait_for_gpus() {
    echo "Waiting for all GPUs to become idle ..."
    while true; do
        BUSY=0
        USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
        for M in $USED; do
            if [ "$M" -ge 2000 ]; then BUSY=1; fi
        done
        for PORT in 8000 8001 8002 8003; do
            if curl -s --max-time 2 "http://localhost:$PORT/health" > /dev/null 2>&1; then
                BUSY=1
            fi
        done
        if [ "$BUSY" -eq 0 ]; then
            echo "All GPUs idle at $(date)."
            sleep 10   # settle before allocating
            return 0
        fi
        echo "  still busy at $(date) (mem MiB: $(echo $USED | tr '\n' ' ')) — retry in 60s"
        sleep 60
    done
}

wait_for_gpus

# ── 1. baseline-noskill ────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " [1/3] baseline-noskill — $(date)"
echo "============================================================"
cd "$SCRIPT_DIR/baseline-noskill"
bash start_vllm.sh
bash launch.sh
bash stop_vllm.sh
echo "[1/3] baseline-noskill finished at $(date)"

sleep 30

# ── 2. text-skill ──────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " [2/3] text-skill — $(date)"
echo "============================================================"
cd "$SCRIPT_DIR/text-skill"
bash start_vllm.sh
bash launch.sh
bash stop_vllm.sh
echo "[2/3] text-skill finished at $(date)"

# give the GPUs a moment to fully release before the next deployment
sleep 30

# ── 3. vis-skill ───────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " [3/3] vis-skill — $(date)"
echo "============================================================"
cd "$SCRIPT_DIR/vis-skill"
bash start_vllm.sh
bash launch.sh
bash stop_vllm.sh
echo "[3/3] vis-skill finished at $(date)"

echo ""
echo "All 27B experiments done at $(date)."
echo "Results:"
echo "  $SCRIPT_DIR/baseline-noskill/results/alfworld/Qwen3.5-27B/noskill_baseline/"
echo "  $SCRIPT_DIR/text-skill/results/alfworld/Qwen3.5-27B/text_baseline/"
echo "  $SCRIPT_DIR/vis-skill/results/alfworld/Qwen3.5-27B/vis_baseline/"
