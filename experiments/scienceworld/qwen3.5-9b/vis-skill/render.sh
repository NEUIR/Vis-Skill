#!/bin/bash
# ============================================================================
# Render SKILL.md skill cards for the ScienceWorld vis-skill CONTROLLED
# experiment. CPU-only (PIL); renderer v6 (structure-preserving, 32px-grid
# aligned). Output: images/skill_cards_v6/<frontmatter-name>.png
# scripts/render_skill_cards.sh renders the cards for all experiments; this
# script is the per-experiment equivalent (idempotent).
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-$(command -v python)}"

"$PYTHON" code/render_skill_cards.py > "$SCRIPT_DIR/render.log" 2>&1

echo "Skill cards rendered -> $SCRIPT_DIR/images/skill_cards_v6 (log: $SCRIPT_DIR/render.log)"
