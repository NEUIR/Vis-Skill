#!/usr/bin/env bash
# Regenerate the main-text figures (Fig. 1, 3, 4, 5). Each plotting script
# reads the result files written by the corresponding analysis run (see
# README.md, "Analyzing Procedural Exposure Bias"), so run those first.
# Fig. 2 is the framework diagram and is not code-generated.
#
# The PNGs are written to $FIG_DIR under the names used in README.md; the
# intermediate PNG/PDF outputs are not kept.
#
#   PYTHON=/path/to/python FIG_DIR=figures bash scripts/make_figures.sh
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-$(command -v python)}"
FIG_DIR="${FIG_DIR:-$REPO_ROOT/figures}"
A="$REPO_ROOT/analysis/alfworld-qwen3.5-9b"
mkdir -p "$FIG_DIR"
TMP_OUT="$(mktemp -d)"
trap 'rm -rf "$TMP_OUT"' EXIT

# Move <dir>/<stem>.png to $FIG_DIR/<name> and drop <dir>/<stem>.pdf.
take() {
    mv "$1/$2.png" "$FIG_DIR/$3"
    rm -f "$1/$2.pdf"
}

# Fig. 1a  -- distractor probability vs. exposure k (length-normalised p_distractor_mean)
LEX="$A/lexical-interference/paper_figures"
"$PYTHON" "$LEX/plot_fig1a_lexical_interference.py" --output_dir "$TMP_OUT" --backbone 9b
take "$TMP_OUT" panel_a_pdistractor_vs_k_9b fig1a_lexical_interference.png

# Fig. 1b  -- per-state Delta p distribution (summed-logP p_distractor_sum)
"$PYTHON" "$LEX/plot_fig1b_delta_p_distribution.py" --output_dir "$TMP_OUT" --panels abc
take "$TMP_OUT" panel_b_delta_p_distribution fig1b_delta_p_distribution.png

# Fig. 1c  -- causal keyword ablation bars
"$PYTHON" "$A/causal-keyword-ablation/paper_figures/plot_fig1c_causal_keyword_ablation.py" --output_dir "$TMP_OUT" --backbone 9b
take "$TMP_OUT" figure2_bars_p_distractor_v4_mean_9b fig1c_causal_keyword_ablation.png

# Fig. 3a  -- rasterized-interference beeswarm of per-state slopes
"$PYTHON" "$A/rasterized-interference/paper_figures/plot_fig3a_rasterized_interference.py" --output_dir "$TMP_OUT" --palette house
take "$TMP_OUT" beeswarm_beta_logp_distractor_9b_house fig3a_rasterized_interference.png

# Fig. 3b  -- cross-model symbol read-out (Qwen3.5-9B + Gemma-4-31B)
"$PYTHON" "$A/mechanism/symbol-encoding/paper_figures/plot_fig3b_symbol_readout.py" --output_dir "$TMP_OUT"
take "$TMP_OUT" fig3_f4_crossmodel_readout fig3b_symbol_readout.png

# Fig. 3c  -- probability-to-behaviour realisation (Qwen3.5-9B + Gemma-4-31B)
"$PYTHON" "$A/probability-to-behavior/paper_figures/plot_fig3c_behavioral_realization.py" --output_dir "$TMP_OUT"
take "$TMP_OUT" figure_behavioral_realization_crossmodel fig3c_behavioral_realization.png

# Fig. 4   -- layer-wise logit-lens exposure slope (norm mode)
"$PYTHON" "$A/mechanism/layerwise-exposure-slope/paper_figures/plot_fig4_logit_lens.py" --output_dir "$TMP_OUT" --mode norm
take "$TMP_OUT" figure_h1_text_norm   fig4_logit_lens_text.png
take "$TMP_OUT" figure_h1_visual_norm fig4_logit_lens_visual.png

# Fig. 5   -- cross-architecture component patching
"$PYTHON" "$A/mechanism/component-patching/paper_figures/plot_fig5_component_patching.py" --output_dir "$TMP_OUT"
take "$TMP_OUT" figure_b_qwen35_9b  fig5_component_patching_qwen.png
take "$TMP_OUT" figure_b_gemma4_31b fig5_component_patching_gemma.png

echo "figures written to $FIG_DIR"
