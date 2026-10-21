#!/bin/bash
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
# =============================================================================
# One-shot driver for the full rasterized-interference run.
#
#   A) offline stages (prepare, render) need NO GPU — the states/documents/
#      structural cards are linked from ../lexical-interference and only the
#      rasterized cards are rendered (~CPU-minutes).
#   B) start the vLLM servers (unless SKIP_VLLM=1).
#   C) smoke test — GATE. If the rasterized measurement chain is broken (cards
#      missing, lossy render, forced stem not a token prefix), stop here
#      instead of burning hours.
#   D) probe (rasterized cells only by default) -> analyze. The
#      analysis merges the existing text/visual cells from
#      ../lexical-interference/results, so the three-condition comparison is
#      complete without re-measuring them.
#
# Skipping work already on disk:
#   SKIP_OFFLINE=1  variants/rasterized already rendered
#   SKIP_VLLM=1     servers already healthy
# probe/analyze are resume-safe on their own (existing cells are skipped).
#
# Usage:  setsid nohup bash run_full.sh > /dev/null 2>&1 &
#         tail -f logs_full_run.log
# =============================================================================
set -uo pipefail

cd "$(cd "$(dirname "$0")" && pwd)"
LOG="logs_full_run.log"
PYTHON="${PYTHON:-$(command -v python)}"
MODEL="${MODELS_DIR:-$REPO_ROOT/models}/Qwen3.5-9B"
LEVELS="${LEVELS:-0,1,2,4,8}"
PORTS="${PORTS:-8000,8001,8002,8003}"

stamp() { echo "[$(date '+%F %T')] $*"; }

fail() {
    stamp "FAILED at $1 — see the stage log. Servers left running for inspection."
    stamp "STATUS=FAILED:$1"
    exit 1
}

{
    stamp "================ rasterized-interference FULL RUN ================"
    stamp "host=$(hostname) levels=$LEVELS ports=$PORTS"

    if [ "${SKIP_OFFLINE:-0}" = "1" ]; then
        stamp "--- A) offline stages SKIPPED (SKIP_OFFLINE=1) ---"
    else
        stamp "--- A) offline stages: prepare, render (no GPU) ---"
        STAGES="prepare,render" bash launch.sh || fail "offline"
    fi
    n_states=$(ls states/*.json 2>/dev/null | grep -vc manifest || true)
    n_raster=$(find variants/rasterized -name 'card*.png' 2>/dev/null | wc -l)
    stamp "inputs on disk: ${n_states} states, ${n_raster} rasterized cards"
    [ "$n_states" -gt 0 ] && [ "$n_raster" -gt 0 ] || fail "inputs-missing"

    if [ "${SKIP_VLLM:-0}" = "1" ]; then
        stamp "--- B) vLLM start SKIPPED (SKIP_VLLM=1) ---"
    else
        stamp "--- B) starting vLLM servers ---"
        bash start_vllm.sh || fail "start_vllm"
    fi
    IFS=',' read -ra PORT_ARR <<< "$PORTS"
    for PORT in "${PORT_ARR[@]}"; do
        curl -s "http://localhost:$PORT/health" > /dev/null 2>&1 || fail "vllm-unhealthy-$PORT"
    done
    stamp "servers healthy"

    stamp "--- C) smoke test (GATE) ---"
    $PYTHON code/smoke_test.py \
        --states_dir states \
        --variants_dir variants \
        --model "$MODEL" \
        --port "${PORT_ARR[0]}" \
        --levels "$LEVELS" \
        --force_template thought_action \
        2>&1 | tee logs_smoke.log || fail "smoke"
    stamp "smoke passed — the rasterized measurement chain is live"

    stamp "--- D) probe + analyze ---"
    STAGES="probe,analyze" PORTS="$PORTS" bash launch.sh || fail "probe/analyze"

    stamp "================ ALL DONE ================"
    stamp "analysis: results/analysis/{rasterized_interference_analysis.json,per_trial.csv}"
    stamp "Fig. 3a:  $PYTHON paper_figures/plot_fig3a_rasterized_interference.py --palette house --per_trial"
    stamp "STATUS=SUCCESS"
} 2>&1 | tee -a "$LOG"
