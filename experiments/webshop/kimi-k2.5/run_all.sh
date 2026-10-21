#!/bin/bash
# Run all 3 experiments for kimi-k2.5 sequentially
# Usage: bash run_all.sh
#
# Required: export DASHSCOPE_API_KEY=<your-key>

set -e
cd "$(dirname "$0")"

PYTHON="${PYTHON:-$(command -v python)}"
if [ ! -x "$PYTHON" ]; then
    echo "[ERROR] $PYTHON not found"
    exit 1
fi

if [ -z "$DASHSCOPE_API_KEY" ]; then
    echo "ERROR: DASHSCOPE_API_KEY not set"
    exit 1
fi

export JAVA_HOME="${JAVA_HOME:-$("$PYTHON" -c 'import jdk4py; print(jdk4py.JAVA_HOME)')}"

echo "============================================"
echo " Model: kimi-k2.5 — ALL 3 experiments"
echo "============================================"

echo ""
echo ">>> [1/3] baseline-noskill"
echo "============================================"
cd baseline-noskill
$PYTHON run_noskill.py --model "kimi-k2.5" --max_workers 20 --max_steps 30 "$@" 2>&1 | tee logs_run.log
cd ..

echo ""
echo ">>> [2/3] text-skill"
echo "============================================"
cd text-skill
$PYTHON run_textskill.py --model "kimi-k2.5" --max_workers 20 --max_steps 30 --use_skill "$@" 2>&1 | tee logs_run.log
cd ..

echo ""
echo ">>> [3/3] vis-skill"
echo "============================================"
cd vis-skill
$PYTHON run_visskill.py --model "kimi-k2.5" --max_workers 20 --max_steps 30 --use_skill "$@" 2>&1 | tee logs_run.log
cd ..

echo ""
echo "============================================"
echo " ALL DONE: kimi-k2.5 (3/3 experiments complete)"
echo "============================================"
