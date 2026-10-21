# Rasterized Interference — modality vs. structure as the source of Visual robustness

## Question

lexical-interference shows that as the distractor-keyword frequency k in the
Skill document increases, **the probability/rank of the distractor action rises
markedly faster under Text, while the Visual (structured skill card) curve is
flatter**. But "Visual" changes two things at once:

1. **modality** — the words are no longer fed as language tokens but as pixels;
2. **structure** — structured layout: titles, sections, hierarchy, indentation.

This experiment adds a third condition, **Rasterized Visual**: **the identical
markdown source document** is drawn line by line as a **plain-text screenshot**
(a single monospace font, a single size, a single colour; no title styling,
no bullets, no code boxes). The three conditions share:

| Condition | Input form | Content |
|---|---|---|
| `text` | characters | the same k-controlled document |
| `rasterized` | pixels (plain-text screenshot) | the same document, character stream preserved verbatim (including `#`, `-`, backticks) |
| `structural` | pixels (structured card) | the same document, markdown rendered as titles/sections/lists/code boxes |

The same decision states, the same candidate actions, the same k = 0,1,2,4,8
and the same forced-decoding measurement path — the only change is the
representation itself.

## Interpretation

For each metric (logP(distractor), P(distractor), margin, entropy…) an OLS
slope over k is fitted and decomposed:

```
total     = slope(text)       − slope(structural)    # the Text−Visual gap of lexical-interference
modality  = slope(text)       − slope(rasterized)    # characters → plain pixels
structure = slope(rasterized) − slope(structural)    # plain pixels → structured pixels
total     = modality + structure                     # holds exactly per state
```

- **Both Rasterized and Structural are clearly flatter than Text**
  (modality_share ≥ 0.7) → robustness comes mainly from the visual modality
  itself weakening direct lexical pull;
- **Rasterized is close to Text and only Structural is more stable**
  (structure_share ≥ 0.7) → structured organisation is what matters;
- **Rasterized lies in between** → both modality and structure contribute.

Each component comes with per-state paired differences and an exact sign test
(`per_state_trends`), plus paired k0→k8 differences (`paired_k0_kmax`) as
mirror evidence that does not assume linearity, and bad-case quadrant analyses
for all three pairwise combinations.

## Layout

```
code/
  render_rasterized.py   # render text variants as plain-text screenshot cards (lossless per-character check)
  run_probe.py           # probe (state × k × rep) cells; by default only the new rasterized cells
  analyze_results.py     # merge local + lexical-interference results; slope decomposition + sign tests + quadrants
  smoke_test.py          # single-state end-to-end check (offline + online)
  probe_lib.py           # inference primitives for the three representations (superset of the lexical-interference probe_lib; measurement path unchanged)
  prompts.py / skill_lexicon.py / render_cards.py / base_config.yaml   # shared with lexical-interference
paper_figures/
  plot_fig3a_rasterized_interference.py  # Fig. 3a (reads results/analysis/per_trial.csv)
states -> ../lexical-interference/states                      # linked by the prepare stage
variants/text   -> ../lexical-interference/variants/text      # the same k documents
variants/visual -> ../lexical-interference/variants/visual    # structured cards (= lexical-interference visual)
variants/rasterized/            # plain-text screenshot cards rendered here
results/                        # newly measured rasterized cells
results/analysis/               # merged analysis
```

## Run

```bash
# Full pipeline (offline render → start servers → smoke gate → probe → analyze)
setsid nohup bash run_full.sh > /dev/null 2>&1 &
tail -f logs_full_run.log

# Step by step
STAGES="prepare,render" bash launch.sh      # no GPU
bash start_vllm.sh                          # 4×TP1 (9B) / 2×TP2 (27B, PORTS=8000,8001 MAX_WORKERS=64)
SMOKE=1 STAGES="probe,analyze" bash launch.sh
bash stop_vllm.sh

# Fig. 3a from the analysis output (results/analysis/per_trial.csv)
python paper_figures/plot_fig3a_rasterized_interference.py --palette house --per_trial
```

- Default `REPS=rasterized`: text/structural cells are merged directly from
  `../lexical-interference/results` (the measurement path is byte-identical,
  so merging is valid; the analyze QA checks that the merged cells agree with
  the states on distractor/candidates).
- `REPS=text,structural,rasterized` re-measures everything in this directory
  (fully self-contained, ~3× the cost).

## Outputs

- `results/analysis/rasterized_interference_analysis.json` — all statistics
  (`trends.slope_decomposition` is the headline result; `per_state_trends` is
  the inferential version)
- `per_trial.csv` / `by_level_rep.csv` / `bad_case_pairs.csv`
