#!/bin/bash
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
# =============================================================================
# Run the three kimi-k2.5 experiments in order: baseline-noskill → text-skill → vis-skill
# Usage: export DASHSCOPE_API_KEY=<your-key> && nohup bash run_all.sh &
# =============================================================================
set -u

export ALFWORLD_DATA="${ALFWORLD_DATA:-$REPO_ROOT/data/alfworld}"

if [ -z "${DASHSCOPE_API_KEY:-}" ]; then
    echo "[ERROR] DASHSCOPE_API_KEY is not set."
    echo "        export DASHSCOPE_API_KEY=<your-key>"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "========== [1/3] baseline-noskill =========="
cd "$SCRIPT_DIR/baseline-noskill" && bash launch.sh
echo ""

echo "========== [2/3] text-skill =========="
cd "$SCRIPT_DIR/text-skill" && bash launch.sh
echo ""

echo "========== [3/3] vis-skill =========="
cd "$SCRIPT_DIR/vis-skill" && bash launch.sh
echo ""

echo "========== ALL DONE =========="
