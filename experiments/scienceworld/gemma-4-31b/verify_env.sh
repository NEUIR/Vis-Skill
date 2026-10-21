#!/bin/bash
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
# =============================================================================
# Sanity checks for the vis-skill-science-gemma environment + Gemma-4 setup.
# Run after create_env.sh. Does NOT touch the GPUs (no model weights loaded).
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_PREFIX="${ENV_PREFIX:-${CONDA_PREFIX:-$REPO_ROOT/envs/vis-skill-science-gemma}}"   # active conda env, else the create_env.sh prefix
PY="$ENV_PREFIX/bin/python"

if [ ! -x "$PY" ]; then
    echo "[ERROR] $PY not found. Run: bash create_env.sh"
    exit 1
fi

echo "== Package versions =="
"$PY" - <<'PYEOF'
import importlib
for m in ("vllm", "transformers", "torch", "accelerate", "scienceworld",
          "openai", "retry", "yaml", "PIL", "psutil", "httpx", "numpy"):
    try:
        mod = importlib.import_module(m)
        print(f"  {m:<14} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        print(f"  {m:<14} IMPORT FAILED: {e}")
PYEOF

echo ""
echo "== vLLM supports Gemma4Unified architecture =="
"$PY" - <<'PYEOF'
from vllm.model_executor.models.registry import ModelRegistry
archs = ModelRegistry.get_supported_archs()
assert "Gemma4UnifiedForConditionalGeneration" in archs, "Gemma4Unified NOT registered in vLLM!"
print("  Gemma4UnifiedForConditionalGeneration: OK")
PYEOF

echo ""
MODEL_DIR="${MODELS_DIR:-$REPO_ROOT/models}/Gemma-4-31B-it"   # downloaded model (see README)
echo "== Chat template of the downloaded model ($MODEL_DIR; enable_thinking off) =="
"$PY" - "$MODEL_DIR" <<'PYEOF'
import sys
from transformers import AutoTokenizer
for model in (sys.argv[1],):
    tok = AutoTokenizer.from_pretrained(model)
    assert tok.chat_template, f"{model}: no chat template found!"
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hello"}]
    out = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                  enable_thinking=False)
    ids = tok.encode(out, add_special_tokens=False)
    assert ids[0] == 2 and ids[1] != 2, f"{model}: expected exactly one leading BOS"
    print(f"  {model}: chat template renders, thinking disabled, single BOS: OK")
PYEOF

echo ""
echo "== java (ScienceWorld JVM) =="
java -version 2>&1 | head -1

echo ""
echo "== Model directories =="
for M in "$MODEL_DIR"; do
    if [ -d "$M" ]; then echo "  $M: OK"; else echo "  $M: MISSING"; fi
done

echo ""
echo "== Shared datasets (bash shared/setup_shared.sh; skill cards: bash scripts/render_skill_cards.sh) =="
for F in test_indices.json science_skill_annotations.json \
         scienceworld_overall_procedure_examples.txt \
         scienceworld_procedure_code_template.py \
         skills_scienceworld skill_cards_v6; do
    P="$SCRIPT_DIR/shared/$F"
    if [ -e "$P" ]; then echo "  $P: OK"
    elif [ "$F" = skill_cards_v6 ]; then echo "  $P: MISSING (bash \$REPO_ROOT/scripts/render_skill_cards.sh)"
    else echo "  $P: MISSING (bash shared/setup_shared.sh)"; fi
done

echo ""
echo "All checks passed."
