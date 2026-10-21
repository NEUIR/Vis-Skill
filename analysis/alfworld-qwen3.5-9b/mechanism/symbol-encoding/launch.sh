#!/bin/bash
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
# =============================================================================
# symbol-encoding on Qwen3.5-9B — all stages in dependency order. Every runner is
# resume-safe (one JSON per cell; existing cells are skipped unless --force),
# so re-running any stage is harmless and only fills gaps.
#
# Usage:
#   bash launch.sh                       # all stages
#   STAGES="f4,analysis" bash launch.sh  # selected stages
# =============================================================================
set -u
cd "$(dirname "$0")"
mkdir -p logs
export PYTORCH_ALLOC_CONF=expandable_segments:True

export MECH_MODEL="${MECH_MODEL:-${MODELS_DIR:-$REPO_ROOT/models}/Qwen3.5-9B}"
PYTHON="${PYTHON:-python}"

STAGES="${STAGES:-f0,f1,f2a,f2b,f3,f4,analysis}"
has_stage() { [[ ",$STAGES," == *",$1,"* ]]; }

# ── F0  zero GPU, uses the existing LI runs ──────────────────────────────
if has_stage f0; then
$PYTHON reanalyze_li_by_tokenization.py 2>/dev/null | tee logs/f0.log
fi

# ── F1  text casing ladder, 5 arms x k{0,4,8} x 165 states  (~2.1 s/cell) ─
if has_stage f1; then
$PYTHON run_text_casing.py --dry-run            # CPU invariant check first
MECH_CUDA_INDEX=0 $PYTHON run_text_casing.py --shard 0 --num-shards 2 > logs/f1_shard0.log 2>&1 &
MECH_CUDA_INDEX=3 $PYTHON run_text_casing.py --shard 1 --num-shards 2 > logs/f1_shard1.log 2>&1 &
wait
fi

# ── F2a  re-render cards with recased keywords (CPU, all arms incl native) ─
# native is re-rendered here too, on purpose: the LI originals were produced in a
# different environment whose box-drawing-glyph font fallback differs. Rendering
# everything here keeps all arms internally comparable. See README.md caveats.
if has_stage f2a; then
$PYTHON render_cased_cards.py --arms native upper altcase --ks 4 8 --workers 32 > logs/f2a_render_k48.log 2>&1
$PYTHON render_cased_cards.py --arms native --ks 0 --workers 16 > logs/f2a_render_k0.log 2>&1
fi

# ── F2b  visual casing, 3 arms x k{0,4,8} x 165 states  (~10 s/cell) ──────
if has_stage f2b; then
MECH_CUDA_INDEX=1 $PYTHON run_visual_casing.py --shard 0 --num-shards 2 > logs/f2b_shard0.log 2>&1 &
MECH_CUDA_INDEX=2 $PYTHON run_visual_casing.py --shard 1 --num-shards 2 > logs/f2b_shard1.log 2>&1 &
wait
fi

# ── F3  case-matched action, match + mismatch arms x k{0,4,8} ─────────────
if has_stage f3; then
$PYTHON run_case_matched_action.py --dry-run --limit 6
MECH_CUDA_INDEX=0 $PYTHON run_case_matched_action.py --shard 0 --num-shards 2 > logs/f3_shard0.log 2>&1 &
MECH_CUDA_INDEX=3 $PYTHON run_case_matched_action.py --shard 1 --num-shards 2 > logs/f3_shard1.log 2>&1 &
wait
fi

# ── F4  modality-fair symbol readout ─────────────────────────────────────
if has_stage f4; then
MECH_CUDA_INDEX=0 $PYTHON run_symbol_readout.py --modality text   --shard 0 --num-shards 2 > logs/f4_text_shard0.log   2>&1 &
MECH_CUDA_INDEX=3 $PYTHON run_symbol_readout.py --modality text   --shard 1 --num-shards 2 > logs/f4_text_shard1.log   2>&1 &
MECH_CUDA_INDEX=1 $PYTHON run_symbol_readout.py --modality visual --shard 0 --num-shards 2 > logs/f4_visual_shard0.log 2>&1 &
MECH_CUDA_INDEX=2 $PYTHON run_symbol_readout.py --modality visual --shard 1 --num-shards 2 > logs/f4_visual_shard1.log 2>&1 &
wait
fi

# ── analysis ─────────────────────────────────────────────────────────────
if has_stage analysis; then
$PYTHON analyze_casing.py       2>/dev/null | tee logs/analyze_casing.log
$PYTHON analyze_case_matched.py 2>/dev/null | tee logs/analyze_case_matched.log
$PYTHON analyze_readout.py      2>/dev/null | tee logs/analyze_readout.log
$PYTHON analyze_single_curve.py 2>/dev/null | tee logs/analyze_single_curve.log
fi
