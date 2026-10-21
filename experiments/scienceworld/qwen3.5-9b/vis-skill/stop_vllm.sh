#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="$ROOT/vllm_server.pid"

if [ -f "$PID_FILE" ]; then
    while read -r pid; do
        kill "$pid" 2>/dev/null && echo "[stop_vllm] Killed PID $pid" || true
    done < "$PID_FILE"
    rm -f "$PID_FILE"
fi

# Also kill any stray vllm processes from this env
pkill -f "vis-skill-science.*vllm.entrypoints" 2>/dev/null || true
echo "[stop_vllm] Done."
