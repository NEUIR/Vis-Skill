# Layer-wise Exposure Slope — where exposure becomes readable action evidence (paper Fig. 4)

## Question

At which layer does repeated exposure to a procedure stop being "text in the
prompt" and start being **action evidence**?

The question is about **formation**: decode the action from the residual
stream after every layer and ask how much that decoding depends on exposure `k`.

## Method

For every (state, k, modality) cell, teacher-force **both** actions and read the
hidden state after each of the 32 LM layers. Apply the model's **own** final
RMSNorm + LM head to the action rows only (a logit lens):

```
s_l(a; k) = Σ_{i ∈ action tokens} log softmax( lens_l(h_{i-1}) )[ a_i ]
M_l(k)    = s_l(a_distractor; k) − s_l(a_correct; k)
```

Per state and layer, fit `M_l(k) = α_l + β_l·k` by OLS over `k ∈ {0,1,2,4,8}`.
The result is 32 β_l values per modality.

Both sides are stored separately and never only the margin, so a fall in M can
never be silently re-read as a rise in the distractor.

### Scoring convention

`logits[i−1]` predicts token `i`; the scored span is `[action_start, T)`, with

```
action_start = len( tokenize( messages + [assistant: FORCED_STEM] ) )
```

i.e. the lexical-interference stem-only tokenisation, **not** character-offset
counting (`lib/position_utils.py::_stem_length` is off by one and must not be used
for scoring). `run_layerwise_slope.py --smoke` asserts `action_start` before any
full run and refuses to proceed on mismatch.

For the **visual** modality, `lib/visual_positions.py` applies the same
stem-only convention in processor/image-pad space and asserts `stem_prefix_ok`
for both forced turns.

### Why this is cheap

No attention weights are needed → sdpa path, no `[H,T,T]` materialisation.
`output_hidden_states=True` costs 33 × [1,T,4096] bf16 ≈ **4.4 GB** at the
visual length T ≈ 16.5k, and the lens is applied to ≤ 12 rows per layer, so the
LM head is never run over the prompt.

## Inference

- Per-state OLS slopes, then averaged. **Never** average per-state ratios.
- **Game-clustered** bootstrap (4000 resamples, seed 0): resample games,
  average within game, then across — the 165 states come from only ~64 games,
  so a game with many states must not dominate.
- Sign test secondary. Effect size first, p-values second.
- Paired Text − Visual per layer over states present in both.

## GATE (hard — nothing downstream is interpretable if it fails)

The lens at l = 31 **is** the model's own final norm + lm_head, so
`s_31(a_d; k)` must equal LI's `logp_distractor_action`. Therefore `β_D[31]`
must reproduce the lexical-interference anchors:

| modality | k ⊆ {0,1,2,4,8} | k ⊆ {0,4,8} |
|---|---|---|
| text   | **0.741** | **0.847** |
| visual | **0.552** | **0.591** |

Tolerance ±0.06. `analyze_layerwise_slope.py` prints PASS / CHECK per modality
and records it in `analysis_k.json["gate"]`. Control: `β_C[31]` must stay near
zero (≈ −0.03); if the correct side is moving, the story is not
"distractor gains evidence".

## Predictions, and what each outcome means

| outcome | reading |
|---|---|
| β_l ≈ 0 for early l, rising monotonically and peaking in late layers | exposure is **integrated with depth** — evidence accumulates rather than being read off at the input. |
| β_l spikes only at the 8 full-attention layers, flat on DeltaNet layers | the effect is carried by softmax attention. |
| β_l spikes on DeltaNet runs and is flat across full-attention layers | evidence that the **linear-attention / mixing pathway** does the accumulation. |
| β_l^text > β_l^visual at every layer, gap widening with depth | representation modulates the accumulation, not just the intake — the Section 4 claim, now localised in depth. |
| β_l^text ≈ β_l^visual despite different final slopes | the modality gap is set **early** (intake/encoding) and merely propagates; the accumulation machinery itself is modality-agnostic. |

The architecture bands in the figure make the third and fourth rows directly
readable: Qwen3.5-9B is 3× Gated DeltaNet → 1× Gated full attention, repeated
8×, so full attention sits at L3, 7, 11, 15, 19, 23, 27, 31 and the other 24
layers are DeltaNet.

## Files

```
README.md                     this file
run_layerwise_slope.py        forwards + logit lens -> results/layerwise-exposure-slope/per_layer/<state>.json
analyze_layerwise_slope.py    per-state OLS, clustered CIs, GATE, band summary
launch.sh                     smoke gate -> run -> analyze
paper_figures/plot_fig4_logit_lens.py   Fig. 4 (reads results/layerwise-exposure-slope/per_layer/)
layerwise_slopes.csv          (generated) per-layer slopes
analysis_k.json               (generated) full statistics
```

## Run

```bash
cd analysis/alfworld-qwen3.5-9b/mechanism/layerwise-exposure-slope
bash launch.sh                       # GPU 0, both modalities, all 165 states
STAGE=analyze bash launch.sh         # re-analyse only, no GPU
LIMIT=5 MODS=text bash launch.sh     # quick debug
```

Cost estimate: 165 states × 5 k × 2 modalities × 2 forced actions = 3300
forwards, half of them visual at T ≈ 16.5k. The visual half dominates; the
text half is cheap (T ≈ 4.8k).

## Caveats, stated up front

1. **A logit lens is not a claim about the model's own computation.** Reading
   the action out of layer l with the final norm + head assumes those are
   approximately the right readout weights at that depth. β_l tells us where
   the exposure-dependent action information becomes *decodable in the final
   basis*, which is the right question for "when does it become evidence", but
   it is not proof that layer l *computes* it. component patching (`../component-patching/`) is
   the causal counterpart.
2. **Only the 32 LM layers are in scope.** The visual skill cards pass through
   a separate vision encoder first. Both modalities enter the same 32-layer LM,
   which is what makes the Text-vs-Visual comparison fair; analysing the vision
   encoder is out of scope.
3. **Linear-attention layers have no head resolution.** β_l is per-layer only;
   the DeltaNet bands can be compared against the full-attention layers but not
   decomposed per head.
4. **k is treated as a continuous regressor** in the OLS fit even though the
   design is `{0,1,2,4,8}`. `logP(k)` is concave (dropping k∈{1,2} *raises* the
   fitted slope from 0.741 to 0.847), so β_l on
   the full grid and on `{0,4,8}` are both reported and must not be mixed in
   one sentence.
