#!/usr/bin/env bash
# component-patching on Qwen3.5-9B — driver.
#
#   STAGE=headline|bands|all     which arms to run (README §8)
#   MODS="text visual"           modalities
#   SHARDS=n                     contiguous shards of the SHARED donor order.
#                                Must be contiguous (the runner enforces it)
#                                because the random-state control is the
#                                previous state in that order; a stride split
#                                would hand every state to a different shard and
#                                leave the control undefined.
#   WAIT_GIB=22                  free VRAM each process needs; the driver polls
#                                until that many cards are actually free, so it
#                                can be left sitting while another job owns the
#                                box. Set WAIT_GIB=0 to bind immediately.
#
# One process per (modality, shard). Order of work is headline first: the two
# Figure (b) arms are what the paper needs, the depth bands are Appendix.
#
# Interpreter: Python 3.12 env (envs/environment_mechanism_py312.yml;
# transformers 5.2.0, torch 2.10.0+cu128); the
# no-intervention reference uses the same backend and precision.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PY="${PY:-${PYTHON:-python}}"
STAGE="${STAGE:-headline}"
MODS="${MODS:-text visual}"
SHARDS="${SHARDS:-2}"
WAIT_GIB="${WAIT_GIB:-22}"
LIMIT="${LIMIT:-}"          # empty = all states; a stray 0 would select zero states
HALF="${HALF:-0}"           # HALF=1 -> contiguous first half of the SHARED donor order

# HALF slices the DONOR ORDER, not the manifest: a contiguous prefix of the seeded
# order is (a) the same 82 states on both models, so the cross-model comparison
# still matches state for state, and (b) preserves the adjacency that the
# random-state control relies on (donor = previous state in this order,
# game-disjoint by construction). A strided or manifest-order subset would break
# (b) or silently compare different states across models.
if [ "$HALF" = "1" ] && [ -z "$LIMIT" ]; then
  N="$("$PY" -c 'import patch_spec as M; print(M.load_donor_map(M.donor_map_path())["n_states"])')"
  LIMIT=$(( (N + 1) / 2 ))
  echo "== HALF=1 -> LIMIT=$LIMIT of $N states (contiguous prefix of the shared donor order)"
fi

export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

free_gib() {  # per-GPU free memory in GiB
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | tr -d ',' | awk -v need="$1" '{ if ($2/1024 >= need) print $1 }'
}

wait_gpus() {  # $1=count $2=need  -> echoes space-separated indices
  local want="$1" need="$2" got
  while :; do
    got="$(free_gib "$need" | head -"$want" | tr '\n' ' ')"
    if [ "$(echo $got | wc -w)" -ge "$want" ]; then echo "$got"; return 0; fi
    echo "  [wait] need $want GPUs with >=${need}GiB free, have $(echo $got | wc -w); retrying in 120s" >&2
    sleep 120
  done
}

NPROC=$(( $(echo "$MODS" | wc -w) * SHARDS ))
echo "== component-patching qwen STAGE=$STAGE mods=[$MODS] shards=$SHARDS -> $NPROC processes"
if [ "$WAIT_GIB" != "0" ]; then
  GPUS="$(wait_gpus "$NPROC" "$WAIT_GIB")"
else
  GPUS="$(seq 0 $((NPROC-1)) | tr '\n' ' ')"
fi
echo "== binding GPUs: $GPUS"

LOG=logs; mkdir -p "$LOG"
pids=(); tags=()
i=0
for mod in $MODS; do
  for s in $(seq 0 $((SHARDS-1))); do
    gpu="$(echo $GPUS | awk -v i=$i '{print $(i+1)}')"
    tag="${mod}_s${s}"
    echo "  [launch] $tag -> cuda:$gpu  ($LOG/expm_qwen_${tag}.log)"
    MECH_CUDA_INDEX="$gpu" "$PY" run_component_patch.py \
        --modality "$mod" --stage "$STAGE" --shard "$s" --num-shards "$SHARDS" \
        ${LIMIT:+--limit "$LIMIT"} \
        > "$LOG/expm_qwen_${tag}.log" 2>&1 &
    pids+=("$!"); tags+=("$tag")
    i=$((i+1)); sleep 6     # stagger the weight loads
  done
done

rc=0
for j in "${!pids[@]}"; do
  if ! wait "${pids[$j]}"; then rc=1; echo "  [FAIL] ${tags[$j]}" >&2; fi
done
[ "$rc" -eq 0 ] || { echo "== one or more shards failed; see $LOG" >&2; exit 1; }
echo "== all qwen shards done (stage=$STAGE)"

echo "== analyze"
"$PY" analyze_component_patch.py --models qwen --stage "$STAGE"
echo "== done"
