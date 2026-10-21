#!/bin/bash
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
# =============================================================================
# Paired env rollouts (correct vs competitor branch) after the probe.
#   text        k = 0,1,2,4,8, all states
#   structural  k = 0,4,8,     all states
# A second pass retries any cell still missing with fresh envs; run_rollout.py
# is resume-safe, so existing records are skipped. The analyze stage then fills
# the rollout fields of the analysis json.
#
# Usage:  bash run_rollout.sh                 # starts and stops the vLLM servers
#         SKIP_VLLM=1 bash run_rollout.sh     # servers already running
# =============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export REPO_ROOT

if [ "${SKIP_VLLM:-0}" != "1" ]; then
    bash start_vllm.sh || { echo "[rollout] FATAL: vLLM not healthy"; exit 1; }
fi

for pass in 1 2; do
    echo "[rollout] $(date +%T) pass $pass: text, k=0,1,2,4,8"
    STAGES=rollout ROLLOUT_REPS=text ROLLOUT_LEVELS=0,1,2,4,8 \
        ROLLOUT_MAX_STATES=-1 ROLLOUT_WORKERS=64 bash launch.sh
    echo "[rollout] $(date +%T) pass $pass: structural, k=0,4,8"
    STAGES=rollout ROLLOUT_REPS=structural ROLLOUT_LEVELS=0,4,8 \
        ROLLOUT_MAX_STATES=-1 ROLLOUT_WORKERS=32 bash launch.sh
done

echo "[rollout] $(date +%T) analyze"
STAGES=analyze bash launch.sh

if [ "${SKIP_VLLM:-0}" != "1" ]; then
    bash stop_vllm.sh
fi
echo "[rollout] $(date +%T) DONE"
