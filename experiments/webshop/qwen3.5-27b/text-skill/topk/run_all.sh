#!/bin/bash
# ============================================================================
# run_all.sh — WebShop text-skill TOP-K series (all 5 levels, one model)
# Usage: nohup bash run_all.sh [9b|27b] > run_all.out 2>&1 &   (model defaults to 27b)
# Levels: top-3 (level-2), top-7 (level+2), top-9 (level+4), top-13 (level+8),
#         top-21 (level+16). level0 (top-5) is the parent text-skill run.
# Servers are started once (mm21 config) and reused across levels.
# ============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_SHORT="${1:-27b}"

for TOP_K in 3 7 9 13 21; do
    echo ""
    echo "############################################################"
    echo "#  text-skill topk — top-$TOP_K ($MODEL_SHORT)  $(date '+%F %T')"
    echo "############################################################"
    bash "$SCRIPT_DIR/launch_level.sh" "$TOP_K" "$MODEL_SHORT"
done

echo ""
echo "=== text-skill topk series finished ($MODEL_SHORT) ==="
for TOP_K in 3 7 9 13 21; do
    case "$MODEL_SHORT" in
        9b)  M=Qwen3.5-9B ;;
        27b) M=Qwen3.5-27B ;;
    esac
    S="$SCRIPT_DIR/results/$M/qwen_webshop_textskill_v1_top${TOP_K}_skill_True/summary.json"
    [ -f "$S" ] && echo "top-$TOP_K: $(python3 -c "import json;d=json.load(open('$S'));print('score=%s sr=%s steps=%s'%(d['score'],d['sr_pct'],d['avg_steps']))")"
done
