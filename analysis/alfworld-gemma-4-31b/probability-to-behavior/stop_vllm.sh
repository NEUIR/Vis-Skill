#!/bin/bash
set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="$SCRIPT_DIR/vllm_pids.txt"
if [ -f "$PID_FILE" ]; then
    while read -r p; do [ -n "$p" ] && kill "$p" 2>/dev/null; done < "$PID_FILE"
    sleep 3
    while read -r p; do [ -n "$p" ] && kill -0 "$p" 2>/dev/null && kill -9 "$p" 2>/dev/null; done < "$PID_FILE"
    rm -f "$PID_FILE"
fi
for PORT in 8000 8001; do
    pid=$(lsof -tiTCP:$PORT -sTCP:LISTEN 2>/dev/null || true)
    [ -n "$pid" ] && kill -9 $pid 2>/dev/null || true
done
