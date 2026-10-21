#!/bin/bash
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
# =============================================================================
# Rasterized-interference experiment — full pipeline.
#
# Why does the representation affect Skill use? Split "Text vs Visual" into two steps:
#
#   MODALITY   text        -> rasterized   (words become pixels; the visible
#                                           character stream is unchanged)
#   STRUCTURE  rasterized  -> structural   (plain pixels gain titles/sections/
#                                           hierarchy/indentation styling)
#
# Everything else is the lexical-interference measurement, unchanged: the same
# decision states, the same k = 0/1/2/4/8 documents, the same candidate action
# sets, the same forced-decoding path. The stages:
#
#   1. prepare  — link the inputs from ../lexical-interference (states, text
#                 variants, structural cards). Nothing is re-extracted or
#                 re-constructed: the three conditions MUST share byte-identical
#                 inputs, and the lexical-interference dir is never written to.
#   2. render   — render every document into rasterized (plain-text screenshot)
#                 cards, with an exact character-fidelity verification
#   3. smoke    — optional single-state end-to-end verification (SMOKE=1)
#   4. probe    — per (state × k × rasterized): free generation + forced
#                 decoding over ALL candidates. Default REPS=rasterized only —
#                 the text and structural cells already exist in
#                 ../lexical-interference/results and are merged at analysis
#                 time. REPS=text,structural,rasterized re-measures everything
#                 locally instead (slower, fully self-contained).
#   5. analyze  — merge local + lexical-interference cells; (k × rep) summary;
#                 slope decomposition total = modality + structure with exact
#                 per-state sign tests; bad-case quadrants per rep pair
#
# Prerequisites:
#   - ../lexical-interference must be complete (states/, variants/text,
#     variants/visual, results/) — this experiment ADDS a condition to it
#   - stage 2 needs no GPU; stages 3-4 need the servers: bash start_vllm.sh
#
# Usage:
#   bash launch.sh                          # prepare,render,probe,analyze
#   STAGES="prepare,render" bash launch.sh  # offline only
#   SMOKE=1 bash launch.sh                  # include the smoke test before probe
#   REPS="text,structural,rasterized" bash launch.sh   # full local re-measure
#   LIMIT=20 bash launch.sh                 # probe only 20 cells (debugging)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-$(command -v python)}"
MODEL="${MODELS_DIR:-$REPO_ROOT/models}/Qwen3.5-9B"
PORTS="${PORTS:-8000,8001,8002,8003}"
MAX_WORKERS="${MAX_WORKERS:-128}"

# ── cache/tmp hygiene (same as ../lexical-interference) ──
RUNTIME_DIR="$SCRIPT_DIR/.runtime"
mkdir -p "$RUNTIME_DIR/cache" "$RUNTIME_DIR/mpl"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$RUNTIME_DIR/cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$RUNTIME_DIR/mpl}"
if [ -z "${TMPDIR:-}" ]; then
    if (echo ok > "/tmp/.ri_preflight_$$" && rm -f "/tmp/.ri_preflight_$$") 2>/dev/null; then
        export TMPDIR=/tmp
    elif (mkdir -p /dev/shm/ri_tmp && echo ok > /dev/shm/ri_tmp/.w && rm -f /dev/shm/ri_tmp/.w) 2>/dev/null; then
        export TMPDIR=/dev/shm/ri_tmp
        echo "[WARN] /tmp not writable (root disk full?) — TMPDIR=$TMPDIR (tmpfs, RAM-backed)"
    else
        echo "[ERROR] neither /tmp nor /dev/shm is writable — cannot run."
        exit 1
    fi
fi

LI_DIR="${LI_DIR:-$(cd "$SCRIPT_DIR/../lexical-interference" && pwd)}"
SKILLS_LINK="$REPO_ROOT/experiments/alfworld/qwen3.5-9b/text-skill/skills_alfworld"

STAGES="${STAGES:-prepare,render,probe,analyze}"
LEVELS="${LEVELS:-0,1,2,4,8}"
REPS="${REPS:-rasterized}"
SMOKE="${SMOKE:-0}"
LIMIT="${LIMIT:--1}"

has_stage() { [[ ",$STAGES," == *",$1,"* ]]; }

if [ ! -x "$PYTHON" ]; then
    echo "[ERROR] $PYTHON not found."
    exit 1
fi

# ── Stage 1: link the shared inputs from lexical-interference ──
if has_stage prepare; then
    echo "=== Stage 1: Prepare inputs (links to $LI_DIR, read-only) ==="
    for sub in states variants/text variants/visual results; do
        if [ ! -e "$LI_DIR/$sub" ]; then
            echo "[ERROR] $LI_DIR/$sub missing — the lexical-interference run must"
            echo "        be complete before this experiment can extend it."
            exit 1
        fi
    done
    # Link the skill corpus and the rendering fonts (idempotent).
    if [ ! -d "$SCRIPT_DIR/skills_alfworld" ] && [ -d "$SKILLS_LINK" ]; then
        ln -sfn "$SKILLS_LINK" "$SCRIPT_DIR/skills_alfworld"
        echo "  linked skills_alfworld -> $SKILLS_LINK"
    fi
    if [ ! -e "$SCRIPT_DIR/code/fonts/DejaVuSansMono.ttf" ]; then
        for F in "$REPO_ROOT/experiments/alfworld/gemma-4-31b/vis-skill/code/fonts"; do
            if [ -e "$F/DejaVuSansMono.ttf" ]; then
                ln -sfn "$(cd "$F" && pwd)" "$SCRIPT_DIR/code/fonts"
                echo "  linked code/fonts -> $F"
                break
            fi
        done
        [ -e "$SCRIPT_DIR/code/fonts/DejaVuSansMono.ttf" ] || {
            echo "[ERROR] no DejaVu fonts reachable — rendering cannot run."; exit 1; }
    fi
    [ -e states ] || ln -s "$LI_DIR/states" states
    mkdir -p variants
    [ -e variants/text ] || ln -s "$LI_DIR/variants/text" variants/text
    [ -e variants/visual ] || ln -s "$LI_DIR/variants/visual" variants/visual
    mkdir -p variants/rasterized results
    n_states=$(ls states/*.json 2>/dev/null | grep -vc manifest || true)
    n_docs=$(find -L variants/text -name 'level_*.md' 2>/dev/null | wc -l)
    n_struct=$(find -L variants/visual -name 'card*.png' 2>/dev/null | wc -l)
    echo "  inputs: ${n_states} states, ${n_docs} documents, ${n_struct} structural cards"
    [ "$n_states" -gt 0 ] && [ "$n_docs" -gt 0 ] && [ "$n_struct" -gt 0 ] || {
        echo "[ERROR] lexical-interference inputs incomplete."; exit 1; }
fi

# ── Stage 2: render rasterized (plain-text screenshot) cards ──
if has_stage render; then
    echo "=== Stage 2: Render rasterized cards ==="
    $PYTHON code/render_rasterized.py \
        --variants_dir "$SCRIPT_DIR/variants" \
        --out_subdir rasterized \
        --parts_glob 'level_*.parts.json' \
        --verify \
        2>&1 | tee logs_render.log
fi

# ── vLLM health check for the inference stages ──
if has_stage probe || [ "$SMOKE" = "1" ]; then
    IFS=',' read -ra PORT_ARR <<< "$PORTS"
    for PORT in "${PORT_ARR[@]}"; do
        if ! curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
            echo "[ERROR] vLLM server on port $PORT is not healthy. Run: bash start_vllm.sh"
            exit 1
        fi
    done
fi

# ── Stage 3 (optional): smoke test on one state ──
if [ "$SMOKE" = "1" ]; then
    echo "=== Stage 3: Smoke test (single state, text + rasterized + structural) ==="
    $PYTHON code/smoke_test.py \
        --states_dir "$SCRIPT_DIR/states" \
        --variants_dir "$SCRIPT_DIR/variants" \
        --model "$MODEL" \
        --port "${PORTS%%,*}" \
        --levels "$LEVELS" \
        2>&1 | tee logs_smoke.log
fi

# ── Stage 4: probe the new condition's cells ──
if has_stage probe; then
    echo "=== Stage 4: Probe (representations: $REPS) ==="
    $PYTHON code/run_probe.py \
        --model "$MODEL" \
        --states_dir "$SCRIPT_DIR/states" \
        --variants_dir "$SCRIPT_DIR/variants" \
        --output_dir "$SCRIPT_DIR/results" \
        --levels "$LEVELS" \
        --representations "$REPS" \
        --force_template thought_action \
        --max_workers "$MAX_WORKERS" \
        --ports "$PORTS" \
        --limit "$LIMIT" \
        2>&1 | tee logs_probe.log
fi

# ── Stage 5: analysis (merges the lexical-interference text/visual cells) ──
if has_stage analyze; then
    echo "=== Stage 5: Analyze results ==="
    $PYTHON code/analyze_results.py \
        --results_dir "$SCRIPT_DIR/results" \
        --merge_results_dirs "$LI_DIR/results" \
        --states_dir "$SCRIPT_DIR/states" \
        --variants_dir "$SCRIPT_DIR/variants" \
        --output_dir "$SCRIPT_DIR/results/analysis" \
        2>&1 | tee logs_analysis.log
fi


echo "=== All requested stages done: $STAGES ==="
