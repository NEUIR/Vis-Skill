#!/bin/bash
# Render the visual skill cards (PNG) for all three benchmarks and place them
# where each Vis-Skill experiment expects them. Rendered images are NOT tracked
# in git; run this once before any Vis-Skill run.
#
#   ALFWorld      render_v5.py           skills_alfworld/    -> images/skill_cards/
#   ScienceWorld  render_skill_cards.py  skills_scienceworld/ -> images/skill_cards_v6/
#   WebShop       render_v7.py           skills/             -> images/skill_cards_v7/
#
# The same card set is shared by every backbone of a benchmark (renderers are
# identical across experiment folders), so we render once and copy.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
E="$REPO_ROOT/experiments"

copy_cards() {  # copy_cards <src_dir> <dst_dir>...
    local src="$1"; shift
    for dst in "$@"; do
        mkdir -p "$dst"; cp -p "$src"/*.png "$dst"/
        echo "  -> ${dst#$REPO_ROOT/} ($(ls "$dst"/*.png | wc -l) cards)"
    done
}

echo "== ALFWorld (render_v5) =="
A="$E/alfworld/gemma-4-31b/vis-skill"
(cd "$A" && "$PYTHON" code/render_v5.py)
copy_cards "$A/images/skill_cards" \
    "$E/alfworld/qwen3.5-9b/vis-skill/images/skill_cards" \
    "$E/alfworld/qwen3.5-27b/vis-skill/images/skill_cards" \
    "$E/alfworld/kimi-k2.5/vis-skill/images/skill_cards"

echo "== ScienceWorld (render_skill_cards v6) =="
S="$E/scienceworld/qwen3.5-9b/vis-skill"
(cd "$S" && "$PYTHON" code/render_skill_cards.py)
copy_cards "$S/images/skill_cards_v6" \
    "$E/scienceworld/qwen3.5-27b/vis-skill/images/skill_cards_v6" \
    "$E/scienceworld/gemma-4-31b/shared/skill_cards_v6" \
    "$E/scienceworld/kimi-k2.5/shared/skill_cards_v6"

echo "== WebShop (render_v7) =="
W="$E/webshop/gemma-4-31b/vis-skill"
(cd "$W" && "$PYTHON" code/render_v7.py)
copy_cards "$W/images/skill_cards_v7" \
    "$E/webshop/qwen3.5-9b/vis-skill/images/skill_cards_v7" \
    "$E/webshop/qwen3.5-27b/vis-skill/images/skill_cards_v7" \
    "$E/webshop/kimi-k2.5/vis-skill/images/skill_cards_v7"
echo "Done."
