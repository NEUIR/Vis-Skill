# Causal Keyword Ablation — is the keyword itself the causal driver?

## Question

lexical-interference shows that the more often a distractor keyword occurs in
the Skill document, the higher the probability of the distractor action, and
that Text is more sensitive than Visual. That is **correlational** evidence:
as k grows the document changes as a whole, so the probability shift could
come from a generic context change rather than from that specific word.

This experiment applies a minimal counterfactual edit to **the same
document** under high exposure (default k = 4 and 8):

| Version | Edit | Purpose |
|---|---|---|
| `original` | none (= the lexical-interference `level_k.md`; the keyword occurs k times) | baseline |
| `counterfactual` | only the distractor keyword is replaced, one word for one word, by a neutral word of the same semantic level (`microwave` → `appliance`, the same mapping used to scrub the base documents) | remove the target word |
| `matched_control` | only one word that occurs **with the same frequency (exactly k times) and is unrelated to the distractor** is replaced by a neutral filler word (similar length, absent from the original document) | rule out "deleting any word changes the probability" |

The three versions have identical sentences, positions, word counts and
layout (asserted per item at construction time: keyword count, unchanged word
count, exactly k differing lines). The control word is chosen by priority and
its tier is recorded:

- **tier 1**: all k occurrences lie inside the k injection sentences
  (frequency and position matched; cleanest);
- **tier 2**: exactly k occurrences in the whole document, at least one inside
  an injection sentence;
- **tier 3**: exactly k occurrences in the whole document.

Excluded words: the keyword itself, the distractor action / phrase / verb, the
correct action, words of the task query, and template words entangled with the
action machinery ("action/command/step/workflow…").

Text and Visual (`structural`, i.e. the Visual Skill cards of the main
experiment) receive the same treatment. `rasterized` (plain-text screenshot)
cards are rendered as well; `REPS=text,structural,rasterized` measures all
three representations together.

## Interpretation (all paired per state, exact sign tests)

```
Δcf = metric(original) − metric(counterfactual)     # effect of removing the keyword
Δmc = metric(original) − metric(matched_control)    # effect of removing the frequency-matched control word
DiD = Δcf − Δmc                                     # causal statistic
```

- **Δcf(logP distractor) significantly > 0 with Δmc ≈ 0 (DiD significant)** →
  the specific word drives the corresponding action prediction, not a generic
  context change;
- **Δcf(text) − Δcf(structural) significantly > 0** → the drop is larger for
  Text, consistent with the visual representation weakening direct lexical pull;
- free generation: `only_original_chose` ≫ `only_counterfactual_chose`
  (McNemar sign test) → the probability drop translates into behaviour.

## Run

```bash
setsid nohup bash run_full.sh > /dev/null 2>&1 &   # full pipeline (with smoke gate)
tail -f logs_full_run.log

# Step by step
STAGES="prepare,construct,render" bash launch.sh   # no GPU
bash start_vllm.sh                                 # 9B: 4×TP1; 27B: PORTS=8000,8001 MAX_WORKERS=64
SMOKE=1 STAGES="probe,analyze" bash launch.sh
bash stop_vllm.sh
```

## Outputs

- `results/analysis/causal_ablation_analysis.json` — Δcf/Δmc/DiD (per state +
  sign tests), cross-representation comparison, behavioural pairing, QA
  (construction assertions and control-word tier distribution)
- `per_trial.csv` / `by_version_rep.csv` / `per_state_deltas.csv`
- `variants/construction_meta.json` — per-state control-word choice, candidate
  pool and all assertions

The paper figure (Fig. 1c) is drawn by
`paper_figures/plot_fig1c_causal_keyword_ablation.py` from `results/analysis/per_trial.csv`.
