#!/bin/bash
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
# =============================================================================
# Start ONLINE vLLM servers for Qwen3.5-9B — one instance per GPU (TP=1).
# Lexical-interference probe: free generation + forced decoding over the
# candidate action set, text AND visual skill conditions.
#
#   - 4 servers: GPU i -> port 8000+i (same layout as the main experiments)
#   - --limit-mm-per-prompt image:6 — a visual probe cell sends the state's
#     5 rendered skill cards (run_probe.py --max_images defaults to 6)
#   - max-model-len 32768 (see MAX_MODEL_LEN below for the measurement) — clean
#     history + the ~2600-word skill document (text) or its 5 skill-card images
#     (visual) + the forced-decoding suffix scored via prompt_logprobs
#
# If servers from another experiment occupy these ports/GPUs, stop them first
# (they are independent processes).
#
# PREFIX_CACHING=1 adds --enable-prefix-caching. The previous deployment ran
# with enable_prefix_caching=False (auto-disabled — Qwen3.5 is a hybrid
# GDN/linear-attention model), which makes the 16 per-candidate
# prompt_logprobs calls re-prefill the whole multimodal prompt every time.
# If the flag is accepted (check "enable_prefix_caching=True" in
# vllm_8000.log), the probe is ~4-5x faster; if the server refuses to start,
# rerun without the flag.
# =============================================================================
set -u

MODEL="${MODELS_DIR:-$REPO_ROOT/models}/Qwen3.5-9B"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-$(command -v python)}"

PORTS=(8000 8001 8002 8003)
GPUS=(0 1 2 3)
GPU_UTIL=0.90
# Measured on real constructed documents: 5 skill cards per visual cell,
# 1000px wide, 2470–3834px tall ≈ 3.1k–4.9k vision tokens each ≈ up to ~21k
# tokens for the cards alone; plus system prompt, clean history and the forced
# suffix this overflows the main runs' 24576 for long-history states. 32768 on
# A100-80GB with the 9B model at gpu-util 0.90 is safe (the scaling experiments
# on the same GPU ran 65536). Raising the cap does not change model behaviour
# for prompts that already fit.
MAX_MODEL_LEN=32768

# ── cache hygiene ──
# vLLM's torch-compile / triton caches default to ~/.cache and /tmp on the
# local root disk, which is shared with model downloads and can fill up.
# XDG_CACHE_HOME redirects VLLM_CACHE_ROOT (~/.cache/vllm) to the NAS.
# TMPDIR must stay local (unix sockets for worker IPC do not work on NFS);
# /tmp writes are tiny, but if the root disk is 100% full the servers will
# fail to start — free space first.
RUNTIME_DIR="$SCRIPT_DIR/.runtime"
mkdir -p "$RUNTIME_DIR/cache" "$RUNTIME_DIR/inductor" "$RUNTIME_DIR/triton" "$RUNTIME_DIR/home"
# HOME override is the important one: several libraries ignore XDG and use
# Path.home() directly (flashinfer → ~/.cache/flashinfer, vllm → ~/.cache/vllm,
# CUDA → ~/.nv). If the root disk is full, their makedirs kills EngineCore at
# startup. Pointing HOME at the NAS makes every HOME-based cache land there.
export HOME="${LI_HOME_OVERRIDE:-$RUNTIME_DIR/home}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$RUNTIME_DIR/cache}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$RUNTIME_DIR/cache/vllm}"
export FLASHINFER_WORKSPACE_DIR="${FLASHINFER_WORKSPACE_DIR:-$RUNTIME_DIR/cache/flashinfer}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$RUNTIME_DIR/inductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$RUNTIME_DIR/triton}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$RUNTIME_DIR/cache/nv}"
# TMPDIR must be local (unix sockets for worker IPC do not work on NFS, and
# NFS silly-rename breaks TemporaryDirectory cleanup). If /tmp is full, fall
# back to /dev/shm (tmpfs, RAM-backed).
if [ -z "${TMPDIR:-}" ]; then
    if (echo ok > "/tmp/.vllm_preflight_$$" && rm -f "/tmp/.vllm_preflight_$$") 2>/dev/null; then
        export TMPDIR=/tmp
    elif (mkdir -p /dev/shm/li_tmp && echo ok > /dev/shm/li_tmp/.w && rm -f /dev/shm/li_tmp/.w) 2>/dev/null; then
        export TMPDIR=/dev/shm/li_tmp
        echo "[WARN] /tmp not writable (root disk full?) — TMPDIR=$TMPDIR (tmpfs, RAM-backed)"
    else
        echo "[ERROR] neither /tmp nor /dev/shm is writable — vLLM cannot start."
        exit 1
    fi
fi

EXTRA_ARGS=()
if [ "${PREFIX_CACHING:-0}" = "1" ]; then
    EXTRA_ARGS+=(--enable-prefix-caching)
    echo "[INFO] prefix caching requested (verify it is not auto-disabled in the logs)"
fi

if [ ! -x "$PYTHON" ]; then
    echo "[ERROR] $PYTHON not found."
    exit 1
fi

PID_FILE="$SCRIPT_DIR/vllm_pids.txt"
: > "$PID_FILE"

echo "Starting ${#PORTS[@]} vLLM instances for $MODEL ..."
for i in "${!PORTS[@]}"; do
    PORT=${PORTS[$i]}
    GPU=${GPUS[$i]}
    LOG="$SCRIPT_DIR/vllm_${PORT}.log"
    echo "  GPU $GPU -> port $PORT (log: $LOG)"
    CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --port "$PORT" \
        --tensor-parallel-size 1 \
        --max-model-len "$MAX_MODEL_LEN" \
        --gpu-memory-utilization "$GPU_UTIL" \
        --no-enable-log-requests \
        --limit-mm-per-prompt '{"image": 6}' \
        ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
        > "$LOG" 2>&1 &
    echo $! >> "$PID_FILE"
    echo "  PID: $!"
done

echo ""
echo "Waiting for servers to be ready (first startup loads weights, be patient)..."
for PORT in "${PORTS[@]}"; do
    WAITED=0
    while ! curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; do
        sleep 5
        WAITED=$((WAITED + 5))
        if [ "$WAITED" -ge 1800 ]; then
            echo "[ERROR] port $PORT not healthy after 1800s — check $SCRIPT_DIR/vllm_${PORT}.log"
            exit 1
        fi
    done
    echo "  Port $PORT ready."
done
echo "All vLLM servers are up!"
