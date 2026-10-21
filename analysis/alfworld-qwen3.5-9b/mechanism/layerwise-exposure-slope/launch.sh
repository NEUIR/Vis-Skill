#!/usr/bin/env bash
# Layer-wise exposure slope (logit lens, paper Fig. 4).
#
# No vLLM: this experiment needs hidden states, so it runs through HuggingFace
# directly with attn_implementation="sdpa".
#
# Usage:
#   bash launch.sh                  # smoke gate -> full run -> analyze
#   STAGE=analyze bash launch.sh    # analysis only, no GPU needed
#   GPU=2 bash launch.sh            # pick the CUDA device
#   LIMIT=5 bash launch.sh          # 5 states only (debugging)
#   MODS=text bash launch.sh        # text only (visual is the slow half)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PY="${PY:-${PYTHON:-python}}"
GPU="${GPU:-0}"
STAGE="${STAGE:-all}"
KS="${KS:-0 1 2 4 8}"
MODS="${MODS:-text visual}"
LIMIT="${LIMIT:-}"

export TOKENIZERS_PARALLELISM=false
# TMPDIR must never point at NFS (unix sockets / dlopen cleanup break there);
# same mitigation the other experiments use.
export TMPDIR="${TMPDIR:-/dev/shm/mech_k_tmp}"
mkdir -p "$TMPDIR"
export MECH_CUDA_INDEX="$GPU"

if [[ "$STAGE" == "all" ]]; then
  echo "[K] ---- SMOKE GATE (2 states; asserts action_start == text_positions) ----"
  "$PY" run_layerwise_slope.py --smoke --ks $KS --modalities text

  echo "[K] ---- full run: modalities='{$MODS}' k='{$KS}' on GPU $GPU ----"
  "$PY" run_layerwise_slope.py --ks $KS --modalities $MODS ${LIMIT:+--limit "$LIMIT"}
fi

if [[ "$STAGE" == "all" || "$STAGE" == "analyze" ]]; then
  echo "[K] ---- analyze ----"
  "$PY" analyze_layerwise_slope.py --k_subset $KS

fi

echo "[K] done."
