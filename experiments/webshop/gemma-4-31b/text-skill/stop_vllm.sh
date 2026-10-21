#!/bin/bash
# Stop the vLLM servers started by this experiment's start_vllm.sh
# (tracked via vllm_pids.txt). Never kills servers of other experiments.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/vllm_pids.txt"

if [ -f "$PID_FILE" ]; then
    while read -r PID; do
        if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
            echo "Killing vLLM server PID $PID"
            kill "$PID" 2>/dev/null
        fi
    done < "$PID_FILE"
    rm -f "$PID_FILE"
else
    # Fallback: only processes that clearly belong to THIS experiment
    for PORT in 8000 8001; do
        PIDS=$(pgrep -f "vllm.entrypoints.openai.api_server.*Gemma-4-31B-it.*--port $PORT" || true)
        for PID in $PIDS; do
            echo "Killing vLLM server PID $PID (port $PORT)"
            kill "$PID" 2>/dev/null
        done
    done
fi

sleep 10
for PORT in 8000 8001; do
    if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "[WARN] port $PORT still serving — check manually if it belongs to this experiment."
    fi
done
echo "Done."
