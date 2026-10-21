#!/bin/bash
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
# =============================================================================
# Lexical-interference experiment — full pipeline.
#
# keyword-frequency interference → probability change → final wrong behaviour:
#   1. extract  — replay the two main experiments' successful trajectories in
#                 the ALFWorld env; per decision state keep observation history,
#                 correct action, admissible-command candidate set, and choose a
#                 skill-corpus distractor (keyword NOT applicable right now)
#   2. construct— build the k = 0/1/2/4/8 skill documents; the distractor
#                 keyword appears exactly k times, total word count constant
#   3. render   — render every document into per-skill PNG cards (visual cond.)
#   4. smoke    — optional single-state end-to-end verification (SMOKE=1)
#   5. probe    — per (state × k × {text,visual}): free generation + forced
#                 decoding over ALL candidates → P(correct), P(distractor), H
#   6. analyze  — distributions/trends vs k, Skill-word Reproduction Rate,
#                 bad-case quadrants (Text wrong / Visual right ...), paired k0→k8 deltas
#
# Prerequisites:
#   - steps 1-3 need the alfworld env ($ALFWORLD_DATA), NOT the vLLM servers
#   - steps 4-5 need the servers: bash start_vllm.sh
#
# Usage:
#   bash launch.sh                     # all stages
#   STAGES="extract,construct,render" bash launch.sh
#   SMOKE=1 bash launch.sh             # include the smoke test before probe
#   LIMIT=20 bash launch.sh            # probe only 20 cells (debugging)
#   MAX_GAMES=10 bash launch.sh        # replay only 10 games (debugging)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

export ALFWORLD_DATA="${ALFWORLD_DATA:-$REPO_ROOT/data/alfworld}"
PYTHON="${PYTHON:-$(command -v python)}"
MODEL="${MODELS_DIR:-$REPO_ROOT/models}/Qwen3.5-9B"

# ── cache/tmp hygiene ──
# The machine's root disk (/tmp, ~/.cache) can fill up (it is shared with model
# downloads). Cache writes go to a NAS-local .runtime dir; experiment outputs
# go to the experiment dir (NAS too).
# TMPDIR must stay on a LOCAL filesystem: NFS breaks TemporaryDirectory
# cleanup (silly-rename → "Directory not empty" inside textworld/fast_downward
# on every env.reset) and unix sockets (vLLM worker IPC) do not work on NFS.
# If /tmp is not writable (root disk full), fall back to /dev/shm (tmpfs).
RUNTIME_DIR="$SCRIPT_DIR/.runtime"
mkdir -p "$RUNTIME_DIR/cache" "$RUNTIME_DIR/mpl"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$RUNTIME_DIR/cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$RUNTIME_DIR/mpl}"
if [ -z "${TMPDIR:-}" ]; then
    if (echo ok > "/tmp/.li_preflight_$$" && rm -f "/tmp/.li_preflight_$$") 2>/dev/null; then
        export TMPDIR=/tmp
    elif (mkdir -p /dev/shm/li_tmp && echo ok > /dev/shm/li_tmp/.w && rm -f /dev/shm/li_tmp/.w) 2>/dev/null; then
        export TMPDIR=/dev/shm/li_tmp
        echo "[WARN] /tmp not writable (root disk full?) — TMPDIR=$TMPDIR (tmpfs, RAM-backed)"
    else
        echo "[ERROR] neither /tmp nor /dev/shm is writable — cannot run."
        exit 1
    fi
fi

MAIN_ROOT="$REPO_ROOT/experiments/alfworld/qwen3.5-9b"
TEXT_RESULTS="${TEXT_RESULTS:-$MAIN_ROOT/text-skill/results/alfworld/Qwen3.5-9B/text_baseline}"
VIS_RESULTS="${VIS_RESULTS:-$MAIN_ROOT/vis-skill/results/alfworld/Qwen3.5-9B/vis_baseline}"
SKILLS_DIR="${SKILLS_DIR:-$MAIN_ROOT/text-skill/skills_alfworld}"
# per-instance top-5 skill ranking used by the main runs (code/ has no copy)
SKILL_RANKING="${SKILL_RANKING:-$MAIN_ROOT/text-skill/code/alfworld_skill_ranking_per_instance.json}"

STAGES="${STAGES:-extract,construct,render,probe,analyze}"
LEVELS="${LEVELS:-0,1,2,4,8}"
SMOKE="${SMOKE:-0}"
LIMIT="${LIMIT:--1}"
MAX_GAMES="${MAX_GAMES:--1}"

has_stage() { [[ ",$STAGES," == *",$1,"* ]]; }

if [ ! -x "$PYTHON" ]; then
    echo "[ERROR] $PYTHON not found."
    exit 1
fi

# ── Stage 1: extract decision states (env replay, needs ALFWORLD_DATA) ──
if has_stage extract; then
    echo "=== Stage 1: Extract decision states ==="
    $PYTHON code/extract_decision_states.py \
        --text_results_dir "$TEXT_RESULTS" \
        --vis_results_dir "$VIS_RESULTS" \
        --output_dir "$SCRIPT_DIR/states" \
        --skills_dir "$SKILLS_DIR" \
        --skill_ranking_path "$SKILL_RANKING" \
        --config "$SCRIPT_DIR/code/base_config.yaml" \
        --split eval_out_of_distribution \
        --require both \
        --replay_source text \
        --max_states_per_traj 3 \
        --max_candidates 16 \
        --min_candidates 5 \
        --max_games "$MAX_GAMES" \
        2>&1 | tee logs_extract.log
fi

# ── Stage 2: construct k-controlled skill documents ──
if has_stage construct; then
    echo "=== Stage 2: Construct k = $LEVELS skill variants ==="
    $PYTHON code/construct_variants.py \
        --states_dir "$SCRIPT_DIR/states" \
        --skills_dir "$SKILLS_DIR" \
        --output_dir "$SCRIPT_DIR/variants" \
        --levels "$LEVELS" \
        --slot_order spread \
        2>&1 | tee logs_construct.log
fi

# ── Stage 3: render visual skill cards ──
if has_stage render; then
    echo "=== Stage 3: Render visual skill cards ==="
    if [ ! -e "$SCRIPT_DIR/code/fonts/DejaVuSansMono.ttf" ]; then
        F="$REPO_ROOT/experiments/alfworld/gemma-4-31b/vis-skill/code/fonts"
        if [ -e "$F/DejaVuSansMono.ttf" ]; then
            ln -sfn "$F" "$SCRIPT_DIR/code/fonts"
            echo "  linked code/fonts -> $F"
        fi
    fi
    $PYTHON code/render_cards.py \
        --variants_dir "$SCRIPT_DIR/variants" \
        --verify \
        2>&1 | tee logs_render.log
fi

# ── vLLM health check for the inference stages ──
if has_stage probe || [ "$SMOKE" = "1" ]; then
    # Preflight: local writable TMPDIR (resolved above; /tmp or /dev/shm).
    if ! (echo ok > "$TMPDIR/.li_preflight_$$" && rm -f "$TMPDIR/.li_preflight_$$") 2>/dev/null; then
        echo "[ERROR] TMPDIR=$TMPDIR is not writable — the vLLM servers and the"
        echo "        env replay cannot run without local tmp space."
        exit 1
    fi
    for PORT in 8000 8001 8002 8003; do
        if ! curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
            echo "[ERROR] vLLM server on port $PORT is not healthy. Run: bash start_vllm.sh"
            exit 1
        fi
    done
fi

# ── Stage 4 (optional): smoke test on one state ──
if [ "$SMOKE" = "1" ]; then
    echo "=== Stage 4: Smoke test (single state, text + visual) ==="
    $PYTHON code/smoke_test.py \
        --states_dir "$SCRIPT_DIR/states" \
        --variants_dir "$SCRIPT_DIR/variants" \
        --model "$MODEL" \
        --port 8000 \
        --levels "$LEVELS" \
        2>&1 | tee logs_smoke.log
fi

# ── Stage 5: probe every (state × k × representation) cell ──
if has_stage probe; then
    echo "=== Stage 5: Probe (free generation + forced decoding) ==="
    $PYTHON code/run_probe.py \
        --model "$MODEL" \
        --states_dir "$SCRIPT_DIR/states" \
        --variants_dir "$SCRIPT_DIR/variants" \
        --output_dir "$SCRIPT_DIR/results" \
        --levels "$LEVELS" \
        --representations "text,visual" \
        --include_no_skill_control \
        --force_template thought_action \
        --max_workers 128 \
        --ports "8000,8001,8002,8003" \
        --limit "$LIMIT" \
        2>&1 | tee logs_probe.log
fi

# ── Stage 6: analysis ──
if has_stage analyze; then
    echo "=== Stage 6: Analyze results ==="
    $PYTHON code/analyze_results.py \
        --results_dir "$SCRIPT_DIR/results" \
        --states_dir "$SCRIPT_DIR/states" \
        --variants_dir "$SCRIPT_DIR/variants" \
        --output_dir "$SCRIPT_DIR/results/analysis" \
        2>&1 | tee logs_analysis.log
fi


echo "=== All requested stages done: $STAGES ==="
