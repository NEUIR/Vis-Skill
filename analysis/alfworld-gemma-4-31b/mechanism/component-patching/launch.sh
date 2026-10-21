#!/usr/bin/env bash
# component-patching on Gemma-4-31B-it — driver.
#
#   STAGE=headline|bands|all     which arms to run (README §8)
#   MODS="text visual"           modalities
#   SOFT=280                     vision token budget arm; 280 matches the
#                                lexical-interference / causal-keyword-ablation /
#                                rasterized-interference / probability-to-behavior
#                                probes (the anchor component-patching gates against). 1120
#                                matches symbol-encoding. Never pooled.
#   SHARDS=n                     contiguous shards of the SHARED donor order
#                                (same rule and same map as the Qwen run).
#   MODE=single|auto             single: one whole checkpoint per process on its
#                                own card (needs >=62GiB free on that card);
#                                auto: shard across every visible card capped at
#                                what is ACTUALLY free — for when another job
#                                owns most of each card. auto forces SHARDS=1.
#
# The 58 GiB bf16 checkpoint is the reason MODE exists: when other jobs hold
# most of each card a whole-card load OOMs, so auto shards the model. If the box
# is idle, single + SHARDS=2 is faster and lets the two modalities run
# concurrently.
#
# Analysis is NOT run here. analyze_component_patch.py reads BOTH models' results, so it runs
# once, from the Qwen tree's component-patching directory (its defaults cover both models).
#
# Interpreter: the Gemma env (envs/environment_scienceworld_gemma.yml;
# transformers 5.15.0 + torch 2.13.0).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PY="${PY:-${PYTHON:-python}}"
STAGE="${STAGE:-headline}"
MODS="${MODS:-text visual}"
SOFT="${SOFT:-280}"
SHARDS="${SHARDS:-2}"
MODE="${MODE:-}"
LIMIT="${LIMIT:-}"
HALF="${HALF:-0}"           # HALF=1 -> contiguous first half of the SHARED donor order

# Same rule as the Qwen driver: the slice is a contiguous prefix of the seeded
# donor order, so both models see the SAME half and the random-state control
# keeps its game-disjoint donor. Never slice the manifest directly.
if [ "$HALF" = "1" ] && [ -z "$LIMIT" ]; then
  N="$("$PY" -c 'import patch_spec as M; print(M.load_donor_map(M.donor_map_path())["n_states"])')"
  LIMIT=$(( (N + 1) / 2 ))
  echo "== HALF=1 -> LIMIT=$LIMIT of $N states (contiguous prefix of the shared donor order)"
fi

# keep the 31B load and its scratch off NFS: a bf16 load thrashes a network
# homedir
export TMPDIR="${TMPDIR:-/tmp/component_patching_gemma}"
mkdir -p "$TMPDIR"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

free_gib() {
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | tr -d ',' | awk -v need="$1" '{ if ($2/1024 >= need) print $1 }'
}

wait_gpus() {  # $1=count $2=need -> space-separated indices
  local want="$1" need="$2" got
  while :; do
    got="$(free_gib "$need" | head -"$want" | tr '\n' ' ')"
    if [ "$(echo $got | wc -w)" -ge "$want" ]; then echo "$got"; return 0; fi
    echo "  [wait] need $want GPUs with >=${need}GiB free, have $(echo $got | wc -w); retrying in 180s" >&2
    sleep 180
  done
}

NMODS=$(echo "$MODS" | wc -w)
if [ -z "$MODE" ]; then
  # decide from what the box offers right now
  if [ "$(free_gib 62 | wc -l)" -ge "$((NMODS * SHARDS))" ]; then
    MODE=single
  else
    MODE=auto; SHARDS=1
    echo "== no $((NMODS * SHARDS)) cards with >=62GiB free; MODE=auto SHARDS=1" >&2
  fi
fi
echo "== component-patching gemma STAGE=$STAGE soft=$SOFT mods=[$MODS] shards=$SHARDS mode=$MODE"

LOG=logs; mkdir -p "$LOG"
pids=(); tags=()
i=0
if [ "$MODE" = "single" ]; then
  NPROC=$((NMODS * SHARDS))
  GPUS="$(wait_gpus "$NPROC" 62)"
  echo "== binding GPUs: $GPUS"
  for mod in $MODS; do
    for s in $(seq 0 $((SHARDS-1))); do
      gpu="$(echo $GPUS | awk -v i=$i '{print $(i+1)}')"
      tag="${mod}_s${s}"
      echo "  [launch] $tag -> cuda:$gpu  ($LOG/expm_gemma_${tag}.log)"
      MECH_CUDA_INDEX="$gpu" MECH_DEVICE_MAP=single "$PY" \
          run_component_patch.py --modality "$mod" --stage "$STAGE" \
          --soft-tokens "$SOFT" --shard "$s" --num-shards "$SHARDS" \
          ${LIMIT:+--limit "$LIMIT"} \
          > "$LOG/expm_gemma_${tag}.log" 2>&1 &
      pids+=("$!"); tags+=("$tag"); i=$((i+1)); sleep 20   # stagger the 58 GiB loads
    done
  done
else
  [ "$SHARDS" -eq 1 ] || { echo "FATAL: MODE=auto needs every GPU, SHARDS must be 1" >&2; exit 2; }
  for mod in $MODS; do
    tag="${mod}_auto"
    echo "  [launch] $tag sharded over all visible GPUs  ($LOG/expm_gemma_${tag}.log)"
    MECH_DEVICE_MAP=auto "$PY" run_component_patch.py \
        --modality "$mod" --stage "$STAGE" --soft-tokens "$SOFT" \
        ${LIMIT:+--limit "$LIMIT"} \
        > "$LOG/expm_gemma_${tag}.log" 2>&1 &
    pids+=("$!"); tags+=("$tag"); sleep 20
  done
fi

rc=0
for j in "${!pids[@]}"; do
  if ! wait "${pids[$j]}"; then rc=1; echo "  [FAIL] ${tags[$j]}" >&2; fi
done
[ "$rc" -eq 0 ] || { echo "== one or more gemma processes failed; see $LOG" >&2; exit 1; }
echo "== all gemma processes done (stage=$STAGE, soft=$SOFT)"
echo "== now run analyze_component_patch.py from the Qwen tree's component-patching dir"
