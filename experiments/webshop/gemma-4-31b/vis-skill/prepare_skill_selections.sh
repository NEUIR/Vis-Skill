#!/bin/bash
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
# =============================================================================
# Build this experiment's skill_selections.json from the QWEN3.8-MAX ANNOTATIONS
# (webshop_skill_annotations.json): the TOP-5 most relevant skills
# per game, selected by rank — NO model-based skill selection.
#
# This is the controlled replacement for the old flow (which copied the
# selections exported by a finished text-skill run). The text-skill
# experiment now uses the exact same annotation file + top-K rule, so both
# experiments use IDENTICAL skills per task by construction.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DST="$SCRIPT_DIR/skill_selections.json"
ANNOTATIONS="$REPO_ROOT/annotation/outputs/webshop_skill_annotations.json"
PYTHON="${PYTHON:-$(command -v python)}"

if [ ! -f "$ANNOTATIONS" ]; then
    echo "[ERROR] annotations file not found: $ANNOTATIONS"
    exit 1
fi

echo "[prepare] generating skill_selections.json from Qwen3.8-Max annotations (top-5) ..."
"$PYTHON" "$SCRIPT_DIR/code/generate_skill_selections.py"

"$PYTHON" - "$DST" <<'EOF'
import collections
import json
import sys

with open(sys.argv[1]) as f:
    sel = json.load(f)

missing = [str(i) for i in range(200) if str(i) not in sel]
if missing:
    print(f"[ERROR] selections missing for game idx {missing[:10]}... ({len(missing)} total)")
    sys.exit(1)

empty = sum(1 for v in sel.values() if not v.get("relevant_skill_names"))
dist = collections.Counter(len(v["relevant_skill_names"]) for v in sel.values())
names = collections.Counter(n for v in sel.values() for n in v["relevant_skill_names"])

print(f"[prepare] {len(sel)} per-task skill selections verified (all 200 games present)")
print(f"[prepare] {empty} games have an EMPTY skill list -> those games run the")
print(f"[prepare] standard (no-skill) procedure in the vis run.")
print(f"[prepare] skills-per-task distribution: {dict(sorted(dist.items()))}")
print(f"[prepare] skill usage counts:")
for name, cnt in sorted(names.items()):
    print(f"            {name:<40} {cnt}")
EOF
