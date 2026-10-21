#!/bin/bash
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
# =============================================================================
# verify_env.sh — Sanity checks for the vis-skill-webshop-gemma environment
# + Gemma-4-31B setup. Run after create_env.sh, scripts/setup_data.sh and
# scripts/render_skill_cards.sh.
# Does NOT touch the GPUs (no model weights loaded).
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_PREFIX="${ENV_PREFIX:-${CONDA_PREFIX:-$REPO_ROOT/envs/vis-skill-webshop-gemma}}"   # active conda env, else the create_env.sh prefix
PY="$ENV_PREFIX/bin/python"
MODEL_DIR="${MODELS_DIR:-$REPO_ROOT/models}/Gemma-4-31B-it"

if [ ! -x "$PY" ]; then
    echo "[ERROR] $PY not found. Run: bash create_env.sh"
    exit 1
fi

echo "== Package versions =="
"$PY" - <<'PYEOF'
import importlib
for m in ("vllm", "transformers", "torch", "accelerate", "openai", "pyserini",
          "spacy", "thefuzz", "flask", "gym", "bs4", "cleantext", "rank_bm25",
          "rich", "retry", "yaml", "PIL", "numpy"):
    try:
        mod = importlib.import_module(m)
        print(f"  {m:<14} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        print(f"  {m:<14} IMPORT FAILED: {e}")
PYEOF

echo ""
echo "== vLLM supports the Gemma-4-31B architecture =="
"$PY" - <<'PYEOF'
from vllm.model_executor.models.registry import ModelRegistry
archs = ModelRegistry.get_supported_archs()
assert "Gemma4ForConditionalGeneration" in archs, \
    "Gemma4ForConditionalGeneration (Gemma-4-31B) NOT registered in vLLM!"
print("  Gemma4ForConditionalGeneration: OK")
PYEOF

echo ""
echo "== Model dir: config + weight shards complete =="
"$PY" - "$MODEL_DIR" <<'PYEOF'
import json, os, sys
model_dir = sys.argv[1]
with open(os.path.join(model_dir, "config.json")) as f:
    cfg = json.load(f)
assert cfg["architectures"] == ["Gemma4ForConditionalGeneration"], cfg["architectures"]
print(f"  config.json arch: {cfg['architectures'][0]}: OK")
with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
    index = json.load(f)
shards = sorted(set(index["weight_map"].values()))
missing = [s for s in shards if not os.path.exists(os.path.join(model_dir, s))]
if missing:
    print(f"  [MISSING] weight shard(s) not downloaded yet: {missing}")
    print("            vLLM cannot serve the model until the download finishes.")
    sys.exit(1)
print(f"  all {len(shards)} weight shard(s) present: OK")
PYEOF

echo ""
echo "== Model dir tokenizer + chat template (what vLLM loads; official recipe) =="
"$PY" - "$MODEL_DIR" <<'PYEOF'
import os, sys
from transformers import AutoTokenizer
model_dir = sys.argv[1]
for f in ("tokenizer.json", "tokenizer_config.json", "processor_config.json",
          "chat_template.jinja"):
    assert os.path.exists(os.path.join(model_dir, f)), f"missing {f} in model dir"
tok = AutoTokenizer.from_pretrained(model_dir)
assert tok.chat_template, "chat_template.jinja was not picked up!"
msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hello"}]
out = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                              enable_thinking=False)
assert out.startswith("<bos><|turn>system"), out[:80]
print("  chat template renders, enable_thinking=false accepted: OK")
PYEOF

echo ""
echo "== JVM for pyserini (jdk4py) =="
JAVA_RT="$ENV_PREFIX/lib/python3.10/site-packages/jdk4py/java-runtime"
if [ -d "$JAVA_RT" ]; then echo "  $JAVA_RT: OK"; else echo "  MISSING: $JAVA_RT"; exit 1; fi

echo ""
echo "== WebShop env import (spacy patched, JVM = jdk4py like in the real runs) =="
cd "$SCRIPT_DIR/baseline-noskill"
JAVA_HOME="$JAVA_RT" "$PY" - <<'PYEOF'
import os, sys
os.environ['JAVA_HOME'] = os.path.join(
    os.path.dirname(os.path.abspath(sys.executable)),
    '..', 'lib', 'python3.10', 'site-packages', 'jdk4py', 'java-runtime'
)
sys.path.insert(0, ".")
sys.path.insert(0, os.path.abspath("../.."))  # experiments/webshop (shared web_agent_site)
from spacy_patch import patch_spacy
patch_spacy()
from web_agent_site.engine.goal import get_reward  # imports spacy + pyserini/JVM
from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
print("  web_agent_site imports (incl. pyserini JVM): OK")
PYEOF

echo ""
echo "== Data files =="
SHARED_DATA="$SCRIPT_DIR/../data"   # experiments/webshop/data (shared dataset dir)
for f in items_shuffle.json items_ins_v2.json items_human_ins.json \
         webshop_overall_procedure_examples.txt webshop_procedure_code_template.py; do
    if [ ! -e "$SHARED_DATA/$f" ]; then
        echo "  [MISSING] $SHARED_DATA/$f  (run: bash scripts/setup_data.sh)"
        exit 1
    fi
done
echo "  shared data dir ($SHARED_DATA): OK"
for f in web_agent_site/envs/web_agent_text_env.py search_engine/indexes; do
    if [ ! -e "$SCRIPT_DIR/../$f" ]; then
        echo "  [MISSING] shared $SCRIPT_DIR/../$f"
        exit 1
    fi
done
echo "  shared web_agent_site + search_engine ($SCRIPT_DIR/..): OK"
for EXP in baseline-noskill text-skill vis-skill; do
    for f in test_indices.json prompts/system_prompt.py; do
        if [ ! -e "$SCRIPT_DIR/$EXP/$f" ]; then
            echo "  [MISSING] $EXP/$f"
            exit 1
        fi
    done
    echo "  $EXP: OK"
done
N_CARDS=$(ls "$SCRIPT_DIR/vis-skill/images/skill_cards_v7/"*.png 2>/dev/null | wc -l)
N_SKILLS=$(ls "$SCRIPT_DIR/text-skill/skills" 2>/dev/null | wc -l)
echo "  vis-skill rendered cards: $N_CARDS/23, text-skill skills: $N_SKILLS/23"

echo ""
echo "All checks passed."
