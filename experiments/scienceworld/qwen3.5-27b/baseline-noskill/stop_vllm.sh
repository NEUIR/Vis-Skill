#!/bin/bash
# Stop the vLLM server started by start_vllm.sh (port 8000, DP=4).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PORT=8000
PID_FILE="$ROOT/vllm_${PORT}.pid"

if [ -f "$PID_FILE" ]; then
    pid="$(cat "$PID_FILE")"
    # start_vllm.sh uses setsid, so the server leads its own process group
    kill -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
    echo "[stop_vllm] Killed process group $pid"
    rm -f "$PID_FILE"
fi

for _ in $(seq 1 60); do
    curl -s "http://localhost:$PORT/health" > /dev/null 2>&1 || break
    sleep 2
done
echo "[stop_vllm] Done."
