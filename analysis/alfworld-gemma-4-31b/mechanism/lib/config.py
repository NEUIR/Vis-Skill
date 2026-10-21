"""
Gemma-tree copy of `mechanism/lib/config.py`.

Semantically identical to the Qwen version EXCEPT for two things:

  * MODEL_PATH defaults to Gemma-4-31B-it (env-overridable via MECH_MODEL).
  * ANALYSIS_ROOT points at the Qwen tree so the lexical-interference and
    causal-keyword-ablation state and variant directories -- which are
    model-independent by construction -- are reused without copying.
    Rebuilding them here would either duplicate hundreds of MB or, worse,
    silently produce a second version that drifts from Qwen's.

The architecture constants (FULL_ATTN_LAYERS, NUM_HEADS, etc.) are Qwen-3.5-9B
specific and do NOT apply to Gemma-4-31B. The Gemma experiments read the layer
plan from the loaded config at run time (component-patching/
archlib_gemma.py) and never use these values.
"""
from __future__ import annotations
import os

# --- paths ----------------------------------------------------------------
# MECH_ROOT is the Gemma mechanism dir. ANALYSIS_ROOT is NOT derived from
# it because the lexical-interference / causal-keyword-ablation inputs live
# under the Qwen tree.
MECH_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.abspath(os.path.join(MECH_ROOT, "..", "..", ".."))
ANALYSIS_ROOT = os.environ.get(
    "MECH_ANALYSIS_ROOT",
    os.path.abspath(os.path.join(MECH_ROOT, "..", "..", "alfworld-qwen3.5-9b")),
)

LI_ROOT = os.path.join(ANALYSIS_ROOT, "lexical-interference")
LI_STATES_DIR = os.path.join(LI_ROOT, "states")
LI_STATES_MANIFEST = os.path.join(LI_STATES_DIR, "manifest.json")
LI_VARIANTS_DIR = os.path.join(LI_ROOT, "variants")
LI_CODE_DIR = os.path.join(LI_ROOT, "code")
LI_RESULTS_DIR = os.path.join(LI_ROOT, "results")

CKA_ROOT = os.path.join(ANALYSIS_ROOT, "causal-keyword-ablation")
CKA_VARIANTS_DIR = os.path.join(CKA_ROOT, "variants")
CKA_META = os.path.join(CKA_VARIANTS_DIR, "construction_meta.json")

# --- model ---------------------------------------------------------------
MODEL_PATH = os.environ.get("MECH_MODEL", os.path.join(os.environ.get("MODELS_DIR", os.path.join(REPO_ROOT, "models")), "Gemma-4-31B-it"))

# --- architecture constants -----------------------------------------------
# Kept for API compatibility with the Qwen lib only; Gemma-4-31B has 60 layers
# and a different mixer pattern. No Gemma script reads these values.
FULL_ATTN_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31]  # UNUSED on Gemma; do NOT trust
NUM_HEADS = 16                                    # UNUSED on Gemma; do NOT trust
HEAD_DIM = 256                                    # UNUSED on Gemma; do NOT trust
HIDDEN_SIZE = 4096                                # UNUSED on Gemma; do NOT trust

# k-levels with lexical-interference documents.
K_LEVELS = [0, 1, 2, 4, 8]
MC_LEVELS = [4, 8]

FORCED_THOUGHT = "Thought: I will now perform the action."
FORCED_STEM = f"{FORCED_THOUGHT}\nAction:"

# Results land INSIDE the Gemma tree.
RESULTS_ROOT = os.path.join(MECH_ROOT, "results")
ATTN_CACHE_DIR = os.path.join(RESULTS_ROOT, "attention_cache")   # path only; created by the caller if needed
