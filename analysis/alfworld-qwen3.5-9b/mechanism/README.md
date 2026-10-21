# Mechanism experiments

Backbone: `Qwen3_5ForConditionalGeneration` at `models/Qwen3.5-9B` (9B).
Dataset: the **165 ALFWorld decision states** extracted in
`../lexical-interference/states/manifest.json`.
No new controlled documents. We reuse:

- `../lexical-interference/variants/text/<state>/level_{0,1,2,4,8}.md` — keyword
  count = k, word count constant;
- `../lexical-interference/variants/visual/<state>/level_{k}/` — the matching
  Visual Skill cards.

The lexical-interference and causal-keyword-ablation experiments establish
that keyword frequency shifts `log P(distractor)` more under Text than under
Visual, and that removing the specific keyword — but not a frequency-matched
control word — removes the effect. The experiments here ask how the two
modalities encode the keyword and where in the network the exposure effect
becomes action evidence.

| experiment | question | paper |
|---|---|---|
| `symbol-encoding` | is the keyword encoded as a native text symbol or as re-recognised pixels? (casing manipulation, cross-model symbol readout) | Fig. 3b |
| `layerwise-exposure-slope` | at which layer does exposure become readable action evidence? (logit lens per layer) | Fig. 4 |
| `component-patching` | does patching the same components (attention mix / MLP) recover the exposure effect on two architectures? | Fig. 5 |

The Gemma-4-31B counterparts live in
`../../alfworld-gemma-4-31b/mechanism/`.

Run `../lexical-interference` and `../causal-keyword-ablation` first; their
`states/` and `variants/` are the inputs here.

## Model-specific caveats

Qwen3.5-9B is a hybrid: only layers `{3, 7, 11, 15, 19, 23, 27, 31}` use full
softmax attention; the other 24 layers use linear (Gated DeltaNet) attention.
Full-attention layers use GQA (16 query heads, 4 KV heads, head_dim 256).

## Position definitions (shared by every experiment)

Every cell (state, k, representation) is scored under a byte-identical forced
turn `<stem>` + `<action>` where
`stem = "Thought: I will now perform the action.\nAction:"`.

- **q_last_stem** = the last token of `<stem>` under the full-prompt
  tokenization. Its next-token distribution produces the first action token.
- **decision span** = `[q_last_stem, q_last_stem + n_action_tokens − 1]`.
- **first discriminative token** = the earliest position within the action
  where distractor and correct action tokens differ.
- **KW(state, k)** = token indices in the assembled prompt that cover any
  whole-word occurrence of the distractor keyword in the skill body.

## Layout

```
mechanism/
├── README.md
├── METHODS.md                            # paper methods and results for layerwise-exposure-slope and component-patching
├── lib/                                  # shared code, imported by every experiment
│   ├── config.py                         # paths, layer/head selection, constants
│   ├── model_guard.py                    # backbone checks
│   ├── prompts_bridge.py                 # re-export of ../lexical-interference/code/prompts.py
│   ├── cells.py                          # cell enumeration over states × k
│   ├── position_utils.py                 # tokenisation / span helpers (legacy compute_positions disabled)
│   ├── text_positions.py                 # text cells: KW / q / action span, stem-only action_start
│   ├── visual_positions.py               # visual cells: image-pad expansion, same action_start convention
│   └── stats.py                          # sign test / bootstrap / correlation helpers
├── symbol-encoding/                      # README.md, launch.sh, paper_figures/
├── layerwise-exposure-slope/             # README.md, launch.sh, paper_figures/
└── component-patching/                   # README.md, launch.sh, paper_figures/
```

Outputs are written to `results/` (not part of the repository).

## How to run

```bash
cd symbol-encoding               && bash launch.sh
cd ../layerwise-exposure-slope   && bash launch.sh
cd ../component-patching && bash launch.sh
```

See each experiment's `README.md` for the full method, gates and outputs.
