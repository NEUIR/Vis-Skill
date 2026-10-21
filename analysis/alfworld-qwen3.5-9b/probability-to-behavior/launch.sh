#!/bin/bash
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
# =============================================================================
# Probability-to-behavior experiment — full pipeline.
#
# Close the chain: Skill keyword frequency k ↑ → probability of the executable
#   wrong competitor ↑ → free generation actually picks it (the env executes it)
#   → (optional) rollout cost
#
# Key difference from lexical-interference: there the distractor is "not
# executable now"; here the competitor comes from the real admissible_commands —
# executable now, but not the correct next step (e.g. open microwave 1 when
# correct = take apple 1 from countertop 1).
#
# Stages:
#   1. extract        — replay the main experiments' successful trajectories in
#                       the ALFWorld env; per decision state choose an ADMISSIBLE
#                       premature competitor (needs $ALFWORLD_DATA, no GPU)
#   2. construct      — k = 0/1/2/4/8 documents (identical machinery to LI:
#                       competitor keyword appears exactly k times, word count
#                       constant)
#   3. render         — structural skill cards (variants/visual)
#   4. render_raster  — rasterized plain-text-screenshot cards (variants/rasterized)
#   5. smoke          — optional single-state end-to-end verification (SMOKE=1)
#   6. probe          — per (state × k × {text,rasterized,structural}):
#                       candidate probabilities + free generation
#   7. rollout        — OPTIONAL (ROLLOUT=1): env rollouts after correct vs
#                       competitor branch, k ∈ {0,8}, text condition by default
#   8. analyze        — bridge statistics: trends of P(competitor) AND
#                       chose-competitor rate vs k per representation,
#                       probability↔behaviour consistency, rollout cost
#
# Usage:
#   bash launch.sh                          # extract..analyze (no rollout)
#   STAGES="extract,construct,render,render_raster" bash launch.sh   # offline
#   SMOKE=1 bash launch.sh
#   ROLLOUT=1 STAGES="rollout" bash launch.sh
#   bash run_rollout.sh                     # full rollout sweep (text + structural)
#   LIMIT=20 bash launch.sh                 # probe only 20 cells (debugging)
#   MAX_GAMES=10 bash launch.sh             # replay only 10 games (debugging)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

export ALFWORLD_DATA="${ALFWORLD_DATA:-$REPO_ROOT/data/alfworld}"
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
    if (echo ok > "/tmp/.p2b_preflight_$$" && rm -f "/tmp/.p2b_preflight_$$") 2>/dev/null; then
        export TMPDIR=/tmp
    elif (mkdir -p /dev/shm/p2b_tmp && echo ok > /dev/shm/p2b_tmp/.w && rm -f /dev/shm/p2b_tmp/.w) 2>/dev/null; then
        export TMPDIR=/dev/shm/p2b_tmp
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
# Per-instance top-5 skill ranking (qwen3.8-max annotation) used by the main
# experiments; extract_behavior_states.py would otherwise look for it in code/.
SKILL_RANKING="${SKILL_RANKING:-$MAIN_ROOT/text-skill/code/alfworld_skill_ranking_per_instance.json}"

STAGES="${STAGES:-extract,construct,render,render_raster,probe,analyze}"
LEVELS="${LEVELS:-0,1,2,4,8}"
ROLLOUT_LEVELS="${ROLLOUT_LEVELS:-0,8}"
ROLLOUT_REPS="${ROLLOUT_REPS:-text}"
ROLLOUT="${ROLLOUT:-0}"
ROLLOUT_MAX_STATES="${ROLLOUT_MAX_STATES:--1}"
ROLLOUT_WORKERS="${ROLLOUT_WORKERS:-4}"
SMOKE="${SMOKE:-0}"
LIMIT="${LIMIT:--1}"
MAX_GAMES="${MAX_GAMES:--1}"

has_stage() { [[ ",$STAGES," == *",$1,"* ]]; }

if [ ! -x "$PYTHON" ]; then
    echo "[ERROR] $PYTHON not found."
    exit 1
fi

# ── fonts for card rendering (idempotent link) ──
if [ ! -e "$SCRIPT_DIR/code/fonts/DejaVuSansMono.ttf" ]; then
    for F in "$REPO_ROOT/experiments/alfworld/gemma-4-31b/vis-skill/code/fonts"; do
        if [ -e "$F/DejaVuSansMono.ttf" ]; then
            ln -sfn "$(cd "$F" && pwd)" "$SCRIPT_DIR/code/fonts"
            echo "[INFO] linked code/fonts -> $F"
            break
        fi
    done
fi

# ── Stage 1: extract decision states with ADMISSIBLE competitors ──
if has_stage extract; then
    echo "=== Stage 1: Extract behavior states (env replay, needs ALFWORLD_DATA) ==="
    $PYTHON code/extract_behavior_states.py \
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

# ── Stage 3: render structural skill cards ──
if has_stage render; then
    echo "=== Stage 3: Render structural skill cards ==="
    $PYTHON code/render_cards.py \
        --variants_dir "$SCRIPT_DIR/variants" \
        --verify \
        2>&1 | tee logs_render.log
fi

# ── Stage 4: render rasterized (plain-text screenshot) cards ──
if has_stage render_raster; then
    echo "=== Stage 4: Render rasterized cards ==="
    $PYTHON code/render_rasterized.py \
        --variants_dir "$SCRIPT_DIR/variants" \
        --out_subdir rasterized \
        --verify \
        2>&1 | tee logs_render_raster.log
fi

# ── vLLM health check for the inference stages ──
if has_stage probe || has_stage rollout || [ "$SMOKE" = "1" ]; then
    IFS=',' read -ra PORT_ARR <<< "$PORTS"
    for PORT in "${PORT_ARR[@]}"; do
        if ! curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
            echo "[ERROR] vLLM server on port $PORT is not healthy. Run: bash start_vllm.sh"
            exit 1
        fi
    done
fi

# ── Stage 5 (optional): smoke test on one state ──
if [ "$SMOKE" = "1" ]; then
    echo "=== Stage 5: Smoke test (single state, all three representations) ==="
    $PYTHON code/smoke_test.py \
        --states_dir "$SCRIPT_DIR/states" \
        --variants_dir "$SCRIPT_DIR/variants" \
        --model "$MODEL" \
        --port "${PORTS%%,*}" \
        --levels "$LEVELS" \
        2>&1 | tee logs_smoke.log
fi

# ── Stage 6: probe every (state × k × representation) cell ──
if has_stage probe; then
    echo "=== Stage 6: Probe (probabilities + free generation) ==="
    $PYTHON code/run_probe.py \
        --model "$MODEL" \
        --states_dir "$SCRIPT_DIR/states" \
        --variants_dir "$SCRIPT_DIR/variants" \
        --output_dir "$SCRIPT_DIR/results" \
        --levels "$LEVELS" \
        --representations "text,rasterized,structural" \
        --include_no_skill_control \
        --force_template thought_action \
        --max_workers "$MAX_WORKERS" \
        --ports "$PORTS" \
        --limit "$LIMIT" \
        2>&1 | tee logs_probe.log
fi

# ── Stage 7 (optional): rollout the branch actions in the real env ──
if has_stage rollout || [ "$ROLLOUT" = "1" ]; then
    echo "=== Stage 7: Rollout (correct vs competitor branch, k=$ROLLOUT_LEVELS, reps=$ROLLOUT_REPS) ==="
    $PYTHON code/run_rollout.py \
        --model "$MODEL" \
        --states_dir "$SCRIPT_DIR/states" \
        --variants_dir "$SCRIPT_DIR/variants" \
        --output_dir "$SCRIPT_DIR/rollout_results" \
        --config "$SCRIPT_DIR/code/base_config.yaml" \
        --split eval_out_of_distribution \
        --levels "$ROLLOUT_LEVELS" \
        --representations "$ROLLOUT_REPS" \
        --branches "correct,competitor" \
        --max_states "$ROLLOUT_MAX_STATES" \
        --max_workers "$ROLLOUT_WORKERS" \
        --ports "$PORTS" \
        2>&1 | tee -a logs_rollout.log
fi

# ── Stage 8: analysis ──
if has_stage analyze; then
    echo "=== Stage 8: Analyze results ==="
    $PYTHON code/analyze_results.py \
        --results_dir "$SCRIPT_DIR/results" \
        --states_dir "$SCRIPT_DIR/states" \
        --variants_dir "$SCRIPT_DIR/variants" \
        --rollout_dir "$SCRIPT_DIR/rollout_results" \
        --output_dir "$SCRIPT_DIR/results/analysis" \
        2>&1 | tee logs_analysis.log
fi


echo "=== All requested stages done: $STAGES ==="
