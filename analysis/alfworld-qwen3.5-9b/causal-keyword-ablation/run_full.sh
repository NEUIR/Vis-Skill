#!/bin/bash
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
# =============================================================================
# One-shot driver for the full causal-keyword-ablation run.
#
#   A) offline stages (prepare, construct, render) need NO GPU.
#   B) start the vLLM servers (unless SKIP_VLLM=1).
#   C) smoke test — GATE. If the version documents or the forced-decoding chain
#      are broken, stop here instead of burning hours.
#   D) probe -> analyze.
#
# Skipping work already on disk:
#   SKIP_OFFLINE=1  variants/ already constructed and rendered
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
LEVELS="${LEVELS:-4,8}"
REPS="${REPS:-text,structural}"
PORTS="${PORTS:-8000,8001,8002,8003}"

stamp() { echo "[$(date '+%F %T')] $*"; }

fail() {
    stamp "FAILED at $1 — see the stage log. Servers left running for inspection."
    stamp "STATUS=FAILED:$1"
    exit 1
}

{
    stamp "================ causal-keyword-ablation FULL RUN ================"
    stamp "host=$(hostname) levels=$LEVELS reps=$REPS ports=$PORTS"

    if [ "${SKIP_OFFLINE:-0}" = "1" ]; then
        stamp "--- A) offline stages SKIPPED (SKIP_OFFLINE=1) ---"
    else
        stamp "--- A) offline stages: prepare, construct, render (no GPU) ---"
        STAGES="prepare,construct,render" REPS="$REPS" bash launch.sh || fail "offline"
    fi
    n_docs=$(find variants/text -name '*_k*.md' 2>/dev/null | wc -l)
    n_cards=$(find variants/visual variants/rasterized -name 'card*.png' 2>/dev/null | wc -l)
    stamp "inputs on disk: ${n_docs} version documents, ${n_cards} cards"
    [ "$n_docs" -gt 0 ] && [ "$n_cards" -gt 0 ] || fail "inputs-missing"

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
        --representations "$REPS" \
        --force_template thought_action \
        2>&1 | tee logs_smoke.log || fail "smoke"
    stamp "smoke passed — the ablation measurement chain is live"

    stamp "--- D) probe + analyze ---"
    STAGES="probe,analyze" REPS="$REPS" PORTS="$PORTS" bash launch.sh || fail "probe/analyze"

    stamp "================ ALL DONE ================"
    stamp "analysis: results/analysis/per_trial.csv (Fig. 1c: paper_figures/plot_fig1c_causal_keyword_ablation.py --per_trial results/analysis/per_trial.csv)"
    stamp "STATUS=SUCCESS"
} 2>&1 | tee -a "$LOG"
