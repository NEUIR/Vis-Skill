#!/bin/bash
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
# =============================================================================
# Causal-keyword-ablation experiment — full pipeline.
#
# Directly test whether the target word in the Skill drives the wrong action:
#   at high exposure (k = 4 / 8), build three versions of the same document
#     original         the distractor keyword occurs k times (= lexical-interference level_k)
#     counterfactual   only the keyword is replaced by a neutral word at the same
#                      semantic level (microwave -> appliance); sentences,
#                      positions, word count and layout are unchanged
#     matched_control  only an unrelated word with the same frequency (= k times)
#                      is replaced by a neutral word
#   Text and Visual (structural, i.e. the main experiment's Visual Skill) get
#   the same treatment.
#
# Reading: if removing the keyword lowers the distractor probability
# significantly, more so for Text, while removing the matched control word
# leaves it unchanged → the word itself drives the action prediction, not a
# generic change of context.
#
# Stages:
#   1. prepare  — link states + the level_{k} documents from
#                 ../lexical-interference (read-only); link skills + fonts
#   2. construct— build the three versions per (state × k) with assertions
#   3. render   — render structural (+rasterized) cards for every version
#   4. smoke    — optional single-state end-to-end verification (SMOKE=1)
#   5. probe    — per (state × k × version × rep): free generation + forced
#                 decoding over ALL candidates
#   6. analyze  — paired Δcf / Δmc / DiD with exact sign tests, cross-rep
#                 comparison, behavioural discordance
#
# Usage:
#   bash launch.sh                       # all stages
#   STAGES="prepare,construct,render" bash launch.sh   # offline only
#   SMOKE=1 bash launch.sh
#   LEVELS=8 bash launch.sh              # only the k=8 exposure
#   REPS="text,structural,rasterized" bash launch.sh   # add the plain-render arm
#   LIMIT=20 bash launch.sh
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
    if (echo ok > "/tmp/.cka_preflight_$$" && rm -f "/tmp/.cka_preflight_$$") 2>/dev/null; then
        export TMPDIR=/tmp
    elif (mkdir -p /dev/shm/cka_tmp && echo ok > /dev/shm/cka_tmp/.w && rm -f /dev/shm/cka_tmp/.w) 2>/dev/null; then
        export TMPDIR=/dev/shm/cka_tmp
        echo "[WARN] /tmp not writable (root disk full?) — TMPDIR=$TMPDIR (tmpfs, RAM-backed)"
    else
        echo "[ERROR] neither /tmp nor /dev/shm is writable — cannot run."
        exit 1
    fi
fi

LI_DIR="${LI_DIR:-$(cd "$SCRIPT_DIR/../lexical-interference" && pwd)}"
SKILLS_LINK="$REPO_ROOT/experiments/alfworld/qwen3.5-9b/text-skill/skills_alfworld"

STAGES="${STAGES:-prepare,construct,render,probe,analyze}"
LEVELS="${LEVELS:-4,8}"
REPS="${REPS:-text,structural}"
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
    for sub in states variants/text variants/construction_meta.json; do
        if [ ! -e "$LI_DIR/$sub" ]; then
            echo "[ERROR] $LI_DIR/$sub missing — the lexical-interference run must"
            echo "        be complete (its level_k documents are this experiment's"
            echo "        'original' version)."
            exit 1
        fi
    done
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
    mkdir -p variants/text variants/visual variants/rasterized results
    echo "  inputs: $(ls states/*.json | grep -vc manifest) states linked from $LI_DIR"
fi

# ── Stage 2: construct the counterfactual / matched-control versions ──
if has_stage construct; then
    echo "=== Stage 2: Construct counterfactuals (levels $LEVELS) ==="
    $PYTHON code/construct_counterfactuals.py \
        --states_dir "$SCRIPT_DIR/states" \
        --li_variants_dir "$LI_DIR/variants" \
        --output_dir "$SCRIPT_DIR/variants" \
        --levels "$LEVELS" \
        2>&1 | tee logs_construct.log
fi

# ── Stage 3: render cards for every version ──
if has_stage render; then
    echo "=== Stage 3: Render version cards ==="
    STYLES=""
    case ",$REPS," in *,structural,*) STYLES="structural" ;; esac
    case ",$REPS," in *,rasterized,*) STYLES="${STYLES:+$STYLES,}rasterized" ;; esac
    if [ -z "$STYLES" ]; then
        echo "  REPS=$REPS has no image condition — nothing to render"
    else
        $PYTHON code/render_versions.py \
            --variants_dir "$SCRIPT_DIR/variants" \
            --styles "$STYLES" \
            --workers 8 \
            2>&1 | tee logs_render.log
    fi
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

# ── Stage 4 (optional): smoke test on one state ──
if [ "$SMOKE" = "1" ]; then
    echo "=== Stage 4: Smoke test (single state) ==="
    $PYTHON code/smoke_test.py \
        --states_dir "$SCRIPT_DIR/states" \
        --variants_dir "$SCRIPT_DIR/variants" \
        --model "$MODEL" \
        --port "${PORTS%%,*}" \
        --levels "$LEVELS" \
        --representations "$REPS" \
        2>&1 | tee logs_smoke.log
fi

# ── Stage 5: probe every (state × k × version × rep) cell ──
if has_stage probe; then
    echo "=== Stage 5: Probe (versions × representations: $REPS) ==="
    $PYTHON code/run_probe.py \
        --model "$MODEL" \
        --states_dir "$SCRIPT_DIR/states" \
        --variants_dir "$SCRIPT_DIR/variants" \
        --output_dir "$SCRIPT_DIR/results" \
        --levels "$LEVELS" \
        --versions "original,counterfactual,matched_control" \
        --representations "$REPS" \
        --force_template thought_action \
        --max_workers "$MAX_WORKERS" \
        --ports "$PORTS" \
        --limit "$LIMIT" \
        2>&1 | tee logs_probe.log
fi

# ── Stage 6: analysis ──
if has_stage analyze; then
    echo "=== Stage 6: Analyze results ==="
    $PYTHON code/analyze_results.py \
        --results_dir "$SCRIPT_DIR/results" \
        --variants_dir "$SCRIPT_DIR/variants" \
        --output_dir "$SCRIPT_DIR/results/analysis" \
        2>&1 | tee logs_analysis.log
fi


echo "=== All requested stages done: $STAGES ==="
