#!/bin/bash
# Stop the Gemma-4-31B vLLM servers started by ./start_vllm.sh.
# Kills each recorded server's ENTIRE process tree (with TP=2 there are
# EngineCore + worker children) — never touches other experiments' processes.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="$SCRIPT_DIR/vllm_pids.txt"

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

if [ -f "$PID_FILE" ]; then
    while read -r PID; do
        if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
            echo "[stop_vllm] Killing vLLM server PID $PID and its whole process tree"
            kill_tree "$PID"
        fi
    done < "$PID_FILE"
    rm -f "$PID_FILE"
else
    for PORT in 8000 8001; do
        PIDS=$(pgrep -f "vllm.entrypoints.openai.api_server.*Gemma-4-31B-it.*--port $PORT" || true)
        for PID in $PIDS; do
            echo "[stop_vllm] Killing vLLM server PID $PID (port $PORT) and its process tree"
            kill_tree "$PID"
        done
    done
fi

sleep 15
for PORT in 8000 8001; do
    if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "[stop_vllm][WARN] port $PORT still serving — kill it manually if it belongs to this experiment."
    fi
done
echo "[stop_vllm] Done."
