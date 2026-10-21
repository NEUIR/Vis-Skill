#!/bin/bash
# Check the downloaded datasets and link them into the locations the experiment code expects.
#
# Expected layout after downloading from HuggingFace into data/ (see README "Data"):
#   data/alfworld/json_2.1.1/valid_unseen/   ALFWorld eval_out_of_distribution split  -> $ALFWORLD_DATA
#   data/alfworld/logic/                     alfred.pddl, alfred.twl2
#   data/webshop/data/                       items_shuffle.json, items_ins_v2.json, items_human_ins.json,
#                                            webshop_overall_procedure_examples.txt, webshop_procedure_code_template.py
#   data/webshop/search_engine/indexes/      prebuilt Lucene index (pyserini)
# ScienceWorld needs no download (pip package `scienceworld`, see README).
#
# The shared WebShop simulator (experiments/webshop/web_agent_site/) resolves
# ../data and ../search_engine relative to itself, so both are linked into
# experiments/webshop/, which every WebShop backbone reads from.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
W="$REPO_ROOT/data/webshop"
A="$REPO_ROOT/data/alfworld"

missing=0
check() {  # check <-f|-d> <path>
    if [ ! "$1" "$2" ]; then echo "[warn] missing ${2#$REPO_ROOT/} (download it first)"; missing=1; fi
}
for f in items_shuffle.json items_ins_v2.json items_human_ins.json \
         webshop_overall_procedure_examples.txt webshop_procedure_code_template.py; do
    check -f "$W/data/$f"
done
check -d "$W/search_engine/indexes"
check -d "$A/json_2.1.1/valid_unseen"
check -f "$A/logic/alfred.pddl"
check -f "$A/logic/alfred.twl2"

link() {  # link <target> <link_name>
    if [ -e "$2" ] && [ ! -L "$2" ]; then echo "[skip] $2 exists and is not a symlink"; return; fi
    ln -sfn "$1" "$2"; echo "[link] ${2#$REPO_ROOT/} -> $1"
}
link "$W/data" "$REPO_ROOT/experiments/webshop/data"
link "$W/search_engine" "$REPO_ROOT/experiments/webshop/search_engine"
[ "$missing" -eq 0 ] || echo "Some dataset files are missing; see the warnings above."
echo "Done. ALFWorld: export ALFWORLD_DATA=$A (this is also the default)."
