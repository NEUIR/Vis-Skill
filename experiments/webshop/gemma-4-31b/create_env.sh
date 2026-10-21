#!/bin/bash
# =============================================================================
# create_env.sh — Create the conda environment for the WebShop x Gemma-4-31B-it
# experiments from envs/environment_webshop_gemma.yml (conda and pip layers pinned
# to the versions the experiments ran with):
#   $REPO_ROOT/envs/vis-skill-webshop-gemma
#
# Usage:  bash create_env.sh            (override the location with ENV_PREFIX=...)
# Equivalent named env:  conda env create -f envs/environment_webshop_gemma.yml
# =============================================================================
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
ENV_PREFIX="${ENV_PREFIX:-$REPO_ROOT/envs/vis-skill-webshop-gemma}"
ENV_FILE="$REPO_ROOT/envs/environment_webshop_gemma.yml"

if [ -x "$ENV_PREFIX/bin/python" ]; then
    echo "[OK] $ENV_PREFIX already exists — nothing to do."
    echo "     Verify with: bash verify_env.sh"
    exit 0
fi

echo "Creating $ENV_PREFIX from $ENV_FILE ..."
conda env create -p "$ENV_PREFIX" -f "$ENV_FILE"

echo ""
echo "Done. Sanity-check with:  bash verify_env.sh"
