#!/bin/bash
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
# =============================================================================
# Sanity checks for the vis-skill-alf-gemma environment + Gemma-4-31B setup.
# Run after create_env.sh. Does NOT touch the GPUs (no model weights loaded).
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_PREFIX="${ENV_PREFIX:-${CONDA_PREFIX:-$REPO_ROOT/envs/vis-skill-alf-gemma}}"   # active conda env, else the create_env.sh prefix
PY="$ENV_PREFIX/bin/python"

if [ ! -x "$PY" ]; then
    echo "[ERROR] $PY not found. Run: bash create_env.sh"
    exit 1
fi

echo "== Package versions =="
"$PY" - <<'PYEOF'
import importlib
for m in ("vllm", "transformers", "torch", "accelerate", "alfworld", "openai", "retry", "yaml", "PIL"):
    try:
        mod = importlib.import_module(m)
        print(f"  {m:<14} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        print(f"  {m:<14} IMPORT FAILED: {e}")
PYEOF

echo ""
echo "== vLLM supports Gemma4 architectures (31B = Gemma4ForConditionalGeneration) =="
"$PY" - <<'PYEOF'
from vllm.model_executor.models.registry import ModelRegistry
archs = ModelRegistry.get_supported_archs()
assert "Gemma4ForConditionalGeneration" in archs, "Gemma4 (31B) NOT registered in vLLM!"
print("  Gemma4ForConditionalGeneration: OK")
assert "Gemma4UnifiedForConditionalGeneration" in archs, "Gemma4Unified NOT registered in vLLM!"
print("  Gemma4UnifiedForConditionalGeneration: OK")
PYEOF

MODEL_DIR="${MODELS_DIR:-$REPO_ROOT/models}/Gemma-4-31B-it"   # downloaded model (see README)

echo ""
echo "== Chat template of the downloaded model ($MODEL_DIR) =="
"$PY" - "$MODEL_DIR" <<'PYEOF'
import sys
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[1])
assert tok.chat_template, "chat_template.jinja was not picked up!"
msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hello"}]
out = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
assert out.startswith("<bos><|turn>system") and out.endswith("<|channel>thought\n<channel|>"), out[:120]
ids = tok.encode(out, add_special_tokens=False)
assert ids[0] == 2 and ids[1] != 2, "expected exactly one leading BOS"
print("  chat template renders, thinking disabled prefix, single BOS: OK")
PYEOF

echo ""
echo "== ALFWorld dataset (\$ALFWORLD_DATA, default data/alfworld) =="
"$PY" - "${ALFWORLD_DATA:-$REPO_ROOT/data/alfworld}" <<'PYEOF'
import os, sys
base = sys.argv[1]
for p in ("json_2.1.1/valid_unseen", "logic/alfred.pddl", "logic/alfred.twl2"):
    full = os.path.join(base, p)
    assert os.path.exists(full), f"missing: {full}"
    print(f"  {full}: OK")
PYEOF

echo ""
echo "== Gemma-4-31B-it model weights present ($MODEL_DIR) =="
"$PY" - "$MODEL_DIR" <<'PYEOF'
import os, json, sys
base = sys.argv[1]
assert os.path.exists(os.path.join(base, "config.json")), "config.json missing"
with open(os.path.join(base, "model.safetensors.index.json")) as f:
    index = json.load(f)
shards = sorted(set(index["weight_map"].values()))
for s in shards:
    p = os.path.join(base, s)
    assert os.path.exists(p), f"missing shard: {p}"
    print(f"  {p}: {os.path.getsize(p)/1e9:.1f} GB OK")
print(f"  total weight bytes: {index['metadata']['total_size']/1e9:.1f} GB (bf16)")
PYEOF

echo ""
echo "All checks passed."
