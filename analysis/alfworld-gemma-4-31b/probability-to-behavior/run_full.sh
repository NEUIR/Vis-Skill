#!/bin/bash
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
# P2B on Gemma-4-31B-it. Reuses Qwen states/variants (already built).
# Runs: probe(text+rasterized+structural × k∈{0,1,2,4,8}) → analyze.
# No rollouts (skip ROLLOUT stage; would need ALFWorld env + long wall-clock).
set -u
cd "$(cd "$(dirname "$0")" && pwd)"
LOG="logs_full_run.log"

PYTHON="${GEMMA_PY:-${PYTHON:-python}}"
MODEL="${GEMMA_MODEL:-${MODELS_DIR:-$REPO_ROOT/models}/Gemma-4-31B-it}"
PORTS="${PORTS:-8000,8001}"
LEVELS="${LEVELS:-0,1,2,4,8}"
REPS="${REPS:-text,rasterized,structural}"
MAX_WORKERS="${MAX_WORKERS:-32}"

QROOT="$REPO_ROOT/analysis/alfworld-qwen3.5-9b/probability-to-behavior"
STATES_DIR="$QROOT/states"
VARIANTS_DIR="$QROOT/variants"

stamp() { echo "[$(date '+%F %T')] $*"; }
fail()  { stamp "FAILED at $1"; exit 1; }

{
    stamp "======== P2B Gemma-4-31B FULL RUN ========"
    n_states=$(find "$STATES_DIR" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l)
    n_text=$(find -L "$VARIANTS_DIR/text" -name '*.md' 2>/dev/null | wc -l)
    n_vis=$(find -L "$VARIANTS_DIR/visual" -name '*.png' 2>/dev/null | wc -l)
    n_rast=$(find -L "$VARIANTS_DIR/rasterized" -name '*.png' 2>/dev/null | wc -l)
    stamp "inputs: states=$n_states text=$n_text visual=$n_vis rasterized=$n_rast"
    [ "$n_states" -gt 0 ] && [ "$n_text" -gt 0 ] && [ "$n_vis" -gt 0 ] && [ "$n_rast" -gt 0 ] \
        || fail "inputs-missing"

    if [ "${SKIP_VLLM:-0}" = "1" ]; then
        stamp "--- vLLM SKIPPED ---"
    else
        bash start_vllm.sh || fail "vllm-start"
    fi
    for p in ${PORTS//,/ }; do
        code=$(curl -s -o /dev/null -w '%{http_code}' -m 3 "http://localhost:$p/health" || true)
        [ "$code" = "200" ] || fail "vllm-health-$p"
    done

    cd code

    stamp "-- probe (k=$LEVELS reps=$REPS workers=$MAX_WORKERS) --"
    "$PYTHON" run_probe.py \
        --model "$MODEL" \
        --states_dir "$STATES_DIR" \
        --variants_dir "$VARIANTS_DIR" \
        --output_dir "../results" \
        --levels "$LEVELS" \
        --representations "$REPS" \
        --ports "$PORTS" \
        --max_workers "$MAX_WORKERS" \
        --limit -1 || fail "probe"

    stamp "-- analyze --"
    "$PYTHON" analyze_results.py \
        --results_dir "../results" \
        --variants_dir "$VARIANTS_DIR" \
        --output_dir "../results/analysis" || fail "analyze"

    stamp "STATUS=OK"
} 2>&1 | tee -a "$LOG"
