#!/bin/bash
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
# =============================================================================
# symbol-encoding on Gemma-4-31B-it — 4-shard parallel across GPUs 0-3.
# Each shard runs on one A100-80G (~58 GiB bf16 weights + activations at
# LI-scale prompts). Modalities in F4 stay serial (text then visual) so we never
# stack two 58 GiB models on the same card.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p logs
export PYTORCH_ALLOC_CONF=expandable_segments:True

export MECH_MODEL="${MECH_MODEL:-${MODELS_DIR:-$REPO_ROOT/models}/Gemma-4-31B-it}"
PYTHON="${PYTHON:-python}"

STAGES="${STAGES:-tok,f0,f1,f2a,f2b,f3,f4,analysis}"
has_stage() { [[ ",$STAGES," == *",$1,"* ]]; }

NSHARDS="${NSHARDS:-4}"
MECH_GPUS="${MECH_GPUS:-0,1,2,3}"
IFS=',' read -r -a GPU_LIST <<< "$MECH_GPUS"

# Fan a run_*.py script out to NSHARDS shards across GPU_LIST (round-robin).
# Any extra args are placed BEFORE --shard/--num-shards.
fan_out() {
    local logtag="$1"; shift
    local script="$1"; shift
    local pids=() rc=0
    for i in $(seq 0 $((NSHARDS-1))); do
        gpu=${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}
        MECH_CUDA_INDEX=$gpu $PYTHON "$script" "$@" \
            --shard "$i" --num-shards "$NSHARDS" \
            > "logs/${logtag}_s${i}.log" 2>&1 &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p" || rc=$?; done
    return $rc
}

if has_stage tok; then
    if [ "${SKIP_TOK:-0}" != "1" ]; then
        echo "== stage: tok_check =="
        $PYTHON tok_check.py 2>&1 | tee logs/tok_check.log
        rc=${PIPESTATUS[0]}
        if [ "$rc" = "2" ]; then
            echo "[abort] tok_check GATE FAIL — symbol-encoding null on this tokenizer."
            exit 2
        fi
    fi
fi

if has_stage f0; then
    echo "== stage: F0 =="
    $PYTHON reanalyze_li_by_tokenization.py 2>&1 | tee logs/f0.log
fi

if has_stage f1; then
    echo "== stage: F1 text casing (${NSHARDS}-way on ${MECH_GPUS}) =="
    $PYTHON run_text_casing.py --dry-run 2>&1 | tee logs/f1_dry.log
    fan_out f1 run_text_casing.py
fi

if has_stage f2a; then
    echo "== stage: F2a render cards (CPU) =="
    $PYTHON render_cased_cards.py --arms native upper altcase --ks 4 8 --workers 16 > logs/f2a_render_k48.log 2>&1
    $PYTHON render_cased_cards.py --arms native --ks 0 --workers 8 > logs/f2a_render_k0.log 2>&1
fi

if has_stage f2b; then
    echo "== stage: F2b visual casing (${NSHARDS}-way on ${MECH_GPUS}) =="
    fan_out f2b run_visual_casing.py
fi

if has_stage f3; then
    echo "== stage: F3 case-matched action (${NSHARDS}-way on ${MECH_GPUS}) =="
    fan_out f3 run_case_matched_action.py
fi

if has_stage f4; then
    echo "== stage: F4 symbol readout text (${NSHARDS}-way on ${MECH_GPUS}) =="
    fan_out f4_text run_symbol_readout.py --modality text
    echo "== stage: F4 symbol readout visual (${NSHARDS}-way on ${MECH_GPUS}) =="
    fan_out f4_visual run_symbol_readout.py --modality visual
fi

if has_stage analysis; then
    echo "== stage: analysis =="
    $PYTHON analyze_casing.py       2>&1 | tee logs/analyze_casing.log
    $PYTHON analyze_case_matched.py 2>&1 | tee logs/analyze_case_matched.log
    $PYTHON analyze_readout.py      2>&1 | tee logs/analyze_readout.log
    $PYTHON analyze_single_curve.py 2>&1 | tee logs/analyze_single_curve.log
fi
