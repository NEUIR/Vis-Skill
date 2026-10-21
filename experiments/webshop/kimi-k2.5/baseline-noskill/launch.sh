#!/bin/bash
set -e
cd "$(dirname "$0")"
PYTHON="${PYTHON:-$(command -v python)}"
export JAVA_HOME="${JAVA_HOME:-$("$PYTHON" -c 'import jdk4py; print(jdk4py.JAVA_HOME)')}"
if [ -z "$DASHSCOPE_API_KEY" ]; then echo "ERROR: DASHSCOPE_API_KEY not set"; exit 1; fi
$PYTHON run_noskill.py --model "kimi-k2.5" --max_workers 20 --max_steps 30 "$@" 2>&1 | tee logs_run.log
