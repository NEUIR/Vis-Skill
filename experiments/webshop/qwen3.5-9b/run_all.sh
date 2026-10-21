#!/bin/bash
# =============================================================================
# Run the three WebShop x Qwen3.5-9B experiments in order:
#     baseline-noskill -> text-skill -> vis-skill
# Each launch.sh starts its own vLLM servers (ports 8000-8003) with its own
# configuration, so the servers are stopped between experiments.
# Top-k series (optional): bash text-skill/topk/run_all.sh, bash vis-skill/topk/run_all.sh
#
# Usage:
#   nohup bash run_all.sh > run_all.out 2>&1 &
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

stop_vllm() {
    pkill -f "[v]llm.entrypoints.openai.api_server.*Qwen3.5-9B" 2>/dev/null || true
    sleep 8
    pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
    for i in $(seq 1 30); do
        BUSY=false
        for port in 8000 8001 8002 8003; do
            curl -s -m 2 "http://localhost:${port}/health" > /dev/null 2>&1 && BUSY=true
        done
        [ "$BUSY" = false ] && return 0
        sleep 2
    done
    echo "[ERROR] ports 8000-8003 still in use"; exit 1
}

for EXP in baseline-noskill text-skill vis-skill; do
    echo ""
    echo "################ Qwen3.5-9B — $EXP  $(date '+%F %T') ################"
    stop_vllm
    bash "$SCRIPT_DIR/$EXP/launch.sh"
done
stop_vllm
echo "=== Qwen3.5-9B: all three WebShop experiments finished ==="
