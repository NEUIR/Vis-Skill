#!/bin/bash
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
# =============================================================================
# One-shot driver for the full lexical-interference run.
#
# Ordering matters and is the point of this script:
#   A) offline stages (extract, construct, render) need the ALFWorld env but NO
#      GPU — running them BEFORE the servers start keeps 4 A100s from idling for
#      the ~30 min they take.
#   B) start the 4 vLLM servers.
#   C) smoke test — GATE. If the measurement chain is broken (in particular if
#      the forced stem is not a token prefix, which would silently null out
#      logP(action), the primary metric), stop here instead of burning hours.
#   D) probe -> analyze.
#
# Each stage is gated on the previous one succeeding.
#
# NOTE on invoking launch.sh: it resolves its stage list with
# `STAGES="${STAGES:-<all stages>}"`, and `${VAR:-default}` substitutes the
# default when VAR is EMPTY as well as when it is unset. So `STAGES="" bash
# launch.sh` silently runs EVERY stage — which is why the smoke step calls
# smoke_test.py directly instead of going through launch.sh.
#
# Skipping work already on disk:
#   SKIP_OFFLINE=1  states/ + variants/ already built (extract is NOT
#                   resume-safe: it always replays from scratch, so never let it
#                   re-run just to reach a later stage)
#   SKIP_VLLM=1     servers already healthy on 8000-8003
# probe/analyze are resume-safe on their own (existing cells are skipped).
# =============================================================================
set -uo pipefail

cd "$(cd "$(dirname "$0")" && pwd)"
LOG="logs_full_run.log"
PYTHON="${PYTHON:-$(command -v python)}"
MODEL="${MODELS_DIR:-$REPO_ROOT/models}/Qwen3.5-9B"
LEVELS="${LEVELS:-0,1,2,4,8}"

stamp() { echo "[$(date '+%F %T')] $*"; }

fail() {
    stamp "FAILED at $1 — see the stage log. Servers left running for inspection."
    stamp "STATUS=FAILED:$1"
    exit 1
}

{
    stamp "================ lexical-interference FULL RUN ================"
    stamp "host=$(hostname) levels=$LEVELS"

    if [ "${SKIP_OFFLINE:-0}" = "1" ]; then
        stamp "--- A) offline stages SKIPPED (SKIP_OFFLINE=1) ---"
    else
        stamp "--- A) offline stages: extract, construct, render (no GPU) ---"
        STAGES="extract,construct,render" bash launch.sh || fail "offline"
    fi
    n_states=$(ls states/*.json 2>/dev/null | grep -vc manifest || true)
    n_docs=$(find variants/text -name 'level_*.md' 2>/dev/null | wc -l)
    n_cards=$(find variants/visual -name 'card*.png' 2>/dev/null | wc -l)
    stamp "inputs on disk: ${n_states} states, ${n_docs} documents, ${n_cards} cards"
    [ "$n_states" -gt 0 ] && [ "$n_cards" -gt 0 ] || fail "inputs-missing"

    if [ "${SKIP_VLLM:-0}" = "1" ]; then
        stamp "--- B) vLLM start SKIPPED (SKIP_VLLM=1) ---"
    else
        stamp "--- B) starting 4 vLLM servers ---"
        bash start_vllm.sh || fail "start_vllm"
    fi
    for PORT in 8000 8001 8002 8003; do
        curl -s "http://localhost:$PORT/health" > /dev/null 2>&1 || fail "vllm-unhealthy-$PORT"
    done
    stamp "servers healthy"

    stamp "--- C) smoke test (GATE) ---"
    # Called directly, NOT via launch.sh (see the STAGES note above).
    $PYTHON code/smoke_test.py \
        --states_dir states \
        --variants_dir variants \
        --model "$MODEL" \
        --port 8000 \
        --levels "$LEVELS" \
        --force_template thought_action \
        2>&1 | tee logs_smoke.log || fail "smoke"
    stamp "smoke passed — the measurement chain is live"

    stamp "--- D) probe + analyze ---"
    STAGES="probe,analyze" bash launch.sh || fail "probe/analyze"

    stamp "================ ALL DONE ================"
    stamp "analysis: results/analysis/per_trial.csv (Fig. 1a/1b: paper_figures/*.py --per_trial results/analysis/per_trial.csv)"
    stamp "STATUS=SUCCESS"
} 2>&1 | tee -a "$LOG"
