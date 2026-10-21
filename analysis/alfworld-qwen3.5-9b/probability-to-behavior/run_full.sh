#!/bin/bash
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
# =============================================================================
# One-shot driver for the full probability-to-behavior run.
#
#   A) offline stages (extract, construct, render, render_raster) need the
#      ALFWorld env but NO GPU.
#   B) start the vLLM servers (unless SKIP_VLLM=1).
#   C) smoke test — GATE (premise check: the competitor must be admissible;
#      forced decoding + free generation must work in all three representations).
#   D) probe -> analyze.
#   E) ROLLOUT=1 adds the optional end-to-end rollout stage between probe and
#      analyze (expensive: env replay-walk per rollout; bounded by
#      ROLLOUT_MAX_STATES).
#
# Skipping work already on disk:
#   SKIP_OFFLINE=1  states/ + variants/ already built (extract is NOT
#                   resume-safe: it always replays from scratch)
#   SKIP_VLLM=1     servers already healthy
# probe/rollout/analyze are resume-safe on their own.
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
    stamp "================ probability-to-behavior FULL RUN ================"
    stamp "host=$(hostname) levels=$LEVELS ports=$PORTS rollout=${ROLLOUT:-0}"

    if [ "${SKIP_OFFLINE:-0}" = "1" ]; then
        stamp "--- A) offline stages SKIPPED (SKIP_OFFLINE=1) ---"
    else
        stamp "--- A) offline stages: extract, construct, render, render_raster (no GPU) ---"
        STAGES="extract,construct,render,render_raster" bash launch.sh || fail "offline"
    fi
    n_states=$(ls states/*.json 2>/dev/null | grep -vc manifest || true)
    n_docs=$(find variants/text -name 'level_*.md' 2>/dev/null | wc -l)
    n_cards=$(find variants/visual variants/rasterized -name 'card*.png' 2>/dev/null | wc -l)
    stamp "inputs on disk: ${n_states} states, ${n_docs} documents, ${n_cards} cards"
    [ "$n_states" -gt 0 ] && [ "$n_cards" -gt 0 ] || fail "inputs-missing"

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
    stamp "smoke passed — the bridge measurement chain is live"

    stamp "--- D) probe ---"
    STAGES="probe" PORTS="$PORTS" bash launch.sh || fail "probe"

    if [ "${ROLLOUT:-0}" = "1" ]; then
        stamp "--- E) rollout (optional end-to-end stage) ---"
        STAGES="rollout" PORTS="$PORTS" bash launch.sh || fail "rollout"
    fi

    stamp "--- F) analyze ---"
    STAGES="analyze" bash launch.sh || fail "analyze"

    stamp "================ ALL DONE ================"
    stamp "analysis: results/analysis/{per_trial.csv,probability_to_behavior_analysis.json}"
    stamp "Fig. 3c: python paper_figures/plot_fig3c_behavioral_realization.py --qwen_csv results/analysis/per_trial.csv --gemma_csv ../../alfworld-gemma-4-31b/probability-to-behavior/results/analysis/per_trial.csv"
    stamp "STATUS=SUCCESS"
} 2>&1 | tee -a "$LOG"
