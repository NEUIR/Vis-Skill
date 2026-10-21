"""
Shared constants for the mechanism experiments.

Nothing here reaches out to the network or the filesystem — pure paths and
literals. Keep it small so every experiment can `from lib.config import *`
without side effects.
"""
from __future__ import annotations
import os

MECH_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANALYSIS_ROOT = os.path.dirname(MECH_ROOT)
REPO_ROOT = os.path.dirname(os.path.dirname(ANALYSIS_ROOT))

LI_ROOT = os.path.join(ANALYSIS_ROOT, "lexical-interference")
LI_STATES_DIR = os.path.join(LI_ROOT, "states")
LI_STATES_MANIFEST = os.path.join(LI_STATES_DIR, "manifest.json")
LI_VARIANTS_DIR = os.path.join(LI_ROOT, "variants")
LI_CODE_DIR = os.path.join(LI_ROOT, "code")
LI_RESULTS_DIR = os.path.join(LI_ROOT, "results")

CKA_ROOT = os.path.join(ANALYSIS_ROOT, "causal-keyword-ablation")
CKA_VARIANTS_DIR = os.path.join(CKA_ROOT, "variants")
CKA_META = os.path.join(CKA_VARIANTS_DIR, "construction_meta.json")

MODEL_PATH = os.environ.get("MECH_MODEL", os.path.join(os.environ.get("MODELS_DIR", os.path.join(REPO_ROOT, "models")), "Qwen3.5-9B"))

# Qwen3.5-9B text config: 32 layers total; only these are full softmax attention.
FULL_ATTN_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31]
NUM_HEADS = 16
HEAD_DIM = 256
HIDDEN_SIZE = 4096

# LI k-levels we have documents for.
K_LEVELS = [0, 1, 2, 4, 8]
# CKA matched-control levels available.
MC_LEVELS = [4, 8]

# Forced-turn constants (byte-identical to LI).
FORCED_THOUGHT = "Thought: I will now perform the action."
FORCED_STEM = f"{FORCED_THOUGHT}\nAction:"

# Results roots (per-experiment scripts append their own subdir).
RESULTS_ROOT = os.path.join(MECH_ROOT, "results")
ATTN_CACHE_DIR = os.path.join(RESULTS_ROOT, "attention_cache")   # path only; created by the caller if needed
