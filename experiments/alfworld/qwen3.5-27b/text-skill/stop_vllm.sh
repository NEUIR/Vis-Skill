#!/bin/bash
# Stop the vLLM servers started by this experiment's start_vllm.sh.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="$SCRIPT_DIR/vllm_pids.txt"

if [ -f "$PID_FILE" ]; then
    while read -r PID; do
        if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
            echo "Killing vLLM server PID $PID and its children"
            # Kill the entire process group to catch EngineCore/TP workers
            pkill -9 -P "$PID" 2>/dev/null
            kill -9 "$PID" 2>/dev/null
        fi
    done < "$PID_FILE"
    rm -f "$PID_FILE"
else
    for PORT in 8000 8001; do
        PIDS=$(pgrep -f "vllm.entrypoints.openai.api_server.*Qwen3.5-27B.*--port $PORT" || true)
        for PID in $PIDS; do
            echo "Killing vLLM server PID $PID (port $PORT) and its children"
            pkill -9 -P "$PID" 2>/dev/null
            kill -9 "$PID" 2>/dev/null
        done
    done
fi

# Also kill any orphaned vLLM workers on this model
pkill -9 -f "VLLM::Worker" 2>/dev/null || true

sleep 15
for PORT in 8000 8001; do
    if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "[WARN] port $PORT still serving — kill it manually if it belongs to this experiment."
    fi
done
echo "Done."
