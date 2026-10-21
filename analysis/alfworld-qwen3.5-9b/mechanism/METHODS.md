# Paper methods & results — layer-wise exposure slope (logit lens) and component patching

Companion note for

* Fig. 4 — `layerwise-exposure-slope/paper_figures/plot_fig4_logit_lens.py`
  (`figure_h1_{text,visual}_{sum,norm}`, Qwen3.5-9B)
* Fig. 5 — `component-patching/paper_figures/plot_fig5_component_patching.py`
  (`figure_b_qwen35_9b`, `figure_b_gemma4_31b`)

Everything below is checked against the code that produced the figures
(`run_layerwise_slope.py`, `analyze_layerwise_slope.py`, `plot_fig4_logit_lens.py`,
`patch_harness.py`, `analyze_component_patch.py`, `plot_fig5_component_patching.py`); formulas are quoted as
implemented, not as intended.  Gemma-4-31B-it layerwise-exposure-slope numbers quoted below are
reported for comparison only; that run is not part of this repository.

---

## 1. Shared material and notation (both experiments, both models)

* **States.** The 165 ALFWorld decision states of the lexical-interference (LI)
  probe, from 64 games. Each state carries a correct action `a_C` and a
  distractor action `a_D` (the PEB probe's pair). The Gemma tree reads the same
  files through `lib/config.py`, which points at the Qwen tree; the documents
  are model-independent and are never regenerated.
* **Exposures.** Skill-procedure documents repeated `k` times, `k ∈ {0,1,2,4,8}`
  for layerwise-exposure-slope; component-patching uses only the endpoints `k=0` (host) and `k=8` (donor).
  Text renders the document as text; Visual renders it as skill cards
  (Qwen: native raster; Gemma: `max_soft_tokens = 280`, the resolution used by
  the behavioural probes).
* **Forced decoding and the scored span.** Every score is a forced-decoding
  log-probability of a whole action. With
  `STEM = "Thought: I will now perform the action.\nAction:"`,
  `action_start = |tokenize(messages ⊕ STEM)|` (stem-only tokenisation, never
  character offsets). Row `i−1` of the logits predicts token `i`, so the scored
  rows are `[action_start−1, T)` and the scored tokens are
  `ids[action_start : T]`. Each cell asserts `stem_prefix_ok` (the stem is a
  literal token prefix of both forced turns) and, in component-patching, `action_ids_match`
  (the `k=8` and `k=0` action id sequences are identical).
* **Depth.** The only depth coordinate used anywhere is relative depth
  `d = l/(L−1)` (`L=32` Qwen, `L=60` Gemma). Absolute layer numbers never appear
  in the main text.

## 2. layerwise-exposure-slope — how the experiment is run

For every state, every `k`, every modality and both actions, **one forward
pass** (165 × 5 × 2 × 2 = 3300 forwards per model). Hidden states are captured
with our own forward hooks on the decoder layers, **not** with
`output_hidden_states=True`: transformers 5.x replaces `hidden_states[-1]` with
the already-final-normed `last_hidden_state`, which would double-norm the top
layer.

**Logit lens.** For layer `l` and action `a`,

```
s_l(a; k) = Σ_{i ∈ action tokens} log softmax( lens_l( h^{(l)}_{i-1} ) )[ a_i ]
lens_l(h) = W_head · RMSNorm_final(h)              (Qwen3.5-9B)
lens_l(h) = W_head · RMSNorm_final(h),  logits ← tanh(logits/30)·30
                                                (Gemma-4-31B-it, final_logit_softcapping)
```

i.e. the lens is the model's *own* readout (final RMSNorm + tied LM head, plus
the softcap where the config sets one), applied to the raw pre-final-norm
residual after layer `l`. `W_head` is run in fp32 on the sliced rows only.

**Per-layer exposure slope.** Define the action margin
`M_l(k) = s_l(a_D; k) − s_l(a_C; k)` and fit, per state and per layer, an
ordinary-least-squares line over the five exposure levels,

```
M_l(k) = α_l + β_l · k ,      β_l = Σ_k (k−k̄)(M_l(k)−M̄_l) / Σ_k (k−k̄)²
```

and the two one-sided slopes `β^D_l`, `β^C_l` from the same regression applied
to `s_l(a_D;·)` and `s_l(a_C;·)` separately. Both sides are always stored; a
fall in `M` can therefore never be silently re-read as a rise in the
distractor.

**Two estimators, reported separately.**
`pooled` = one OLS on the across-state mean level at each `k`. Because the OLS
slope is a *linear* functional of `y` on a common k-grid, this is identically
the plain state-mean of the per-state slopes (verified numerically on Gemma:
both 2.6462). `per_state_mean` as stored in `analysis_k*.json` is **not** that
plain mean: its point estimate is the game-clustered statistic — equal weight
per *game* (mean over the 64 games of within-game means), e.g. 2.6143 on Gemma
text vs 2.6462 state-weighted. Both are legitimate; they answer "average
state" vs "average game". CIs are game-clustered for both estimators. Anchors
and gates are compared estimator-to-estimator (the vLLM LI anchors are pooled),
and the h1 plot scripts' console gate uses the state-weighted/pooled value;
mixing the conventions is exactly how one manufactures a fake gate failure.

**Uncertainty.** Game-clustered bootstrap, 4000 resamples, seed 0: resample the
64 games with replacement, average within game, then across games; the 2.5/97.5
percentiles give the 95% CI. Plus an exact two-sided sign test per layer.

**Gates (run before any figure).**
1. *Structural:* at `l = L−1` the lens must reproduce the model's own `logits`
   (norm-of-hooked-state vs the model's post-norm hidden state at rounding
   level; fp32-vs-bf16 log-prob residual bounded by bf16 ULP at the softcap
   scale, `max(0.05, 3·ULP)` strict / `max(0.25, 8·ULP)` hard on Gemma).
2. *Behavioural:* `β^D_{L−1}` must match the published LI anchor for the same
   states/k-set/estimator — Qwen 0.741 (text) / 0.552 (visual), tol 0.06 nats;
   Gemma 2.6564 (text) / 1.3609 (visual, pooled vLLM), tol `max(0.10, 10%)`.
   Both models PASS (Qwen 0.789/0.592; Gemma pooled 2.646/1.372).

## 3. How `figure_h1_{text,visual}_{norm,sum}` (Fig. 4) are drawn

Source: the per-layer JSONs of all 165 states (`results/layerwise-exposure-slope/per_layer/`),
**never** the collapsed slope CSV; `plot_fig4_logit_lens.py` aggregates them
itself. A state is
dropped if any of its cells fails `stem_prefix_ok` or has an empty action span.

Cell value, for modality `m`, layer `l`, exposure `k`:

```
H_m[l, k] = mean over states of ( s_l(a_D; k) − s_l(a_D; 0) )
```

with `s_l` the **sum** over action tokens (`mode=sum`) or the **per-action-token
mean**, `s_l / n_action_tokens` (`mode=norm`). The `_norm` files are the
per-token normalisation; `_sum` is the unnormalised action score. **No fitting
enters the heatmap** — it is the raw exposure × layer landscape of the
distractor side, referenced to `k=0`.

Drawing: one figure per modality; `imshow` heatmap, layers on `y` (top = layer
0), `k ∈ {0,1,2,4,8}` on `x`; sequential `Blues`; near-square axes
(`set_box_aspect`-equivalent physical sizing); ticks directly on the axes;
`y` ticks at layer 0 plus the full-attention layers (3,7,…,31). Colour scale is noise-robust:
`vmin = 0`, `vmax` = the **shared 98th percentile of the positive cells across
both modalities** (so Text and Visual are on one comparable scale and a few
deep-layer high-`k` outliers do not wash the map out); values above `vmax` are
clipped into an over-colour. The correct side is **not drawn**: `β^C` stays
≈0 on this shared scale (Qwen max |β^C| = 0.053 nats); it is stated in the
caption instead. Serif/Times, `pdf.fonttype=42`, 450 dpi.

The only regression in these scripts is a **console gate**, not part of the
image: the top-layer OLS slope of the distractor score over `k` is recomputed
from the same matrices and compared to a stored anchor. The script recomputes
the state-weighted mean of per-state slopes (0.791/0.594) and compares it to the
*game-weighted* values from `analysis_k.json` (0.789/0.592) with a 0.02-nat
slack — i.e. it tolerates the weighting-convention difference of §2 and fails
only on a real discrepancy. The figure is only trusted if the script prints
`GATE=PASS`.

## 4. layerwise-exposure-slope results and conclusions

* **Qwen3.5-9B.** `β^M ≈ 0` for `d ≲ 0.1`, gently rising through mid-depth,
  peaking at `d ≈ 0.87` (Text +1.43 nats/exposure) and settling at the readout
  (`β^M_31` Text +0.82, Visual +0.59; gate `β^D_31` 0.789/0.592 game-weighted,
  0.791/0.594 state-weighted, vs LI anchors 0.741/0.552 — PASS either way).
  Visual tracks Text until `d ≈ 0.7` then falls below it: the modality gap is
  written in the last ~5 layers. `β^C` never exceeds 0.053 nats (6.4 % of the
  top-layer margin slope) — the correct action is exposure-flat layer by
  layer. Layer-type means: GDN layers 0.273 (Text) / 0.350 (Visual),
  full-attention 0.493 / 0.521 — both mixer types carry slope.
* **Gemma-4-31B-it.** The slope is ~0 and oscillating through the sliding
  window (mean over the 50 sliding layers: Text +0.061, CI includes 0; Visual
  −0.071), and is concentrated in the 10 full-attention layers
  (Text +0.384 [0.252, 0.512]). The curve rises sharply only in the top decile:
  peak Text +2.81 at `d = 1.00`, Visual +1.46 at `d = 0.98`; gate `β^D_59`
  pooled/state-weighted 2.646 and game-weighted 2.614 vs the vLLM pooled anchor
  2.6564 (visual 1.370/1.361 vs 1.3609) — PASS. Onset (frac-of-final) Text 0.22/0.51/0.98,
  Visual 0.51/0.51/0.95 — i.e. Gemma's evidence is more *late-loaded* than
  Qwen's (75 % of the final slope only at `d ≈ 0.98` vs 0.74). `β^C` reaches
  0.406 nats = 14.5 % of the top-layer margin slope: still small, but the
  "correct side is flat" claim is **weaker on Gemma** and must be stated per
  model.
* **Cross-model (shape only).** Both architectures keep the exposure signal out
  of the residual stream until roughly the second quartile (25 %-of-final at
  `d ≈ 0.22–0.26` for Text) and write most of it in the top decile. Absolute
  magnitudes are not comparable (softcap, vision budget).
* **Conclusion licensed:** *where* the evidence becomes **readable** —
  progressively, late, and on Gemma preferentially in full-attention layers.
  Decodability only: nothing here says any layer *computed* the effect.

## 5. component-patching — how the experiment is run

Three forwards per (state, modality, action), all `use_cache=False`, no KV
cache, no GDN state, no attention weights touched:

1. **donor** — forward `k=8`, recording at the decision rows the output of every
   patch site (token-mixer module and MLP module of each layer; on Gemma also
   the two sandwich-norm outputs, Appendix-only);
2. **host** — forward `k=0`, recording the same sites (this doubles as the
   self-patch control and as the baseline);
3. **patched** — re-run `k=0`, replacing the chosen sites' outputs **at the
   decision rows only** with the donor's, aligned **by action-token index**
   (row predicting action token `s` ← donor row predicting action token `s`),
   then recompute the action score.

Patch point, identically on both architectures: *after the module completes,
before the residual add* (`mix_L{l}` = `linear_attn`/`self_attn` output;
`mlp_L{l}` = MLP output). On Gemma the patched vector subsequently passes
`post_attention_layernorm`/`post_feedforward_layernorm` (sandwich norms Qwen
does not have), which strips its magnitude — bounded by the two Gemma-only
Appendix arms `*_postnorm`. `enable_moe_block = False` and
`hidden_size_per_layer_input = 0` are asserted at load, so `layer.mlp` is the
whole MLP contribution.

**Metric.** Per state and arm,
`G = S_{k=8}(a_D) − S_{k=0}(a_D)`, `R = S_patch(a_D) − S_{k=0}(a_D)`, and

```
Recovered exposure effect = 100 · mean(R) / mean(G)
```

a **ratio of means**, never a mean of per-state ratios; the ratio is recomputed
inside every game-clustered bootstrap sample (4000, seed 0). The correct action
is scored identically and reported as the PEB-specificity control **in nats**
(a ratio against the small `G_C` is unstable by construction).

**Arms (headline).** `all_mix` (every layer's mixer), `all_mlp` (every layer's
MLP), plus controls `embed` (must be exactly 0: identical tokens), `self_all`
(host's own activations; must be exactly 0), `resid_last` (final layer output;
must be exactly 100 %), `rand_all_mix`/`rand_all_mlp` (a **different game's**
`k=8` activations, from one shared `donor_map.json` so both models receive the
same mismatched donors).

**Structural ceiling (proven, README §5.1).** Outside the mixer module both
models are row-wise (residual add, RMSNorm incl. Gemma's sandwich norms, MLP),
and the decision-row embedding is identical between donor and host
(`action_ids_match`, verified by `embed ≡ 0`). Therefore replacing every mixer
output at the decision rows reproduces the donor's row trajectory exactly and
`all_mix ≡ resid_last ≡ 100 %`. Verified to 1e-6 on a toy model and observed as
`+100.0 [100.0, 100.0]` on all four real blocks. `all_mix` is a harness
ceiling, drawn as a reference line, **not** a finding.

## 6. How `figure_b` (Fig. 5) is drawn

Script: `component-patching/paper_figures/plot_fig5_component_patching.py` (reads `results/component-patching/{per_state.csv,analysis.json}`). Output:
`component-patching/paper_figures/figure_b_qwen35_9b.{png,pdf}` and `figure_b_gemma4_31b.{png,pdf}`,
model name in the file name only. Type and canvas copy
`lexical-interference/paper_figures/plot_fig1b_delta_p_distribution.py` verbatim: serif
Times stack, label 13 / tick 11 / legend 10.5, `FIGSIZE = (4.15, 3.85)`,
dpi 600 with `bbox_inches="tight", pad_inches=0.03`.

A bar chart would show nothing here: the mixer value is pinned at 100 % by
construction and the MLP value is one number per modality. The scatter draws
**every state** instead, from `results/component-patching/per_state.csv` (written by
analyze_component_patch.py):

* Axes use short standard terms; the method sentence lives in the caption.
  x = "Exposure effect": that state's preference gap for the distractor
  at k=8 minus k=0 (`G`). y = "Recovered by patch": how much of `G`
  returns when the labelled component's k=8 output is transplanted into the
  k=0 run at the action-scoring rows (`R`). Legend names the transplanted
  component only ("Token mixer", "MLP (text)", "MLP (visual)"); the donor/host
  exposures are stated once in the caption. No per-state ratio is ever formed
  or averaged — the unstable-ratio problem cannot arise on this canvas.
* **Grey cloud = token-mixer swaps.** It collapses onto the diagonal `y = x`,
  state by state: the empirical picture of the structural ceiling (§5). The
  dotted diagonal and the dashed zero line are the only reference marks.
* **Clay / periwinkle clouds = MLP swaps** (skill as text / as image). Their
  position relative to the zero line is the finding: below it on Qwen (the MLP
  opposes the bias), between zero and the diagonal on Gemma (the MLP carries
  ~57 % of it).
* The headline ratio-of-means is deliberately **not** drawn as a fitted line:
  it is a ratio of means, not a per-state regression, and a slope through the
  points would invite the mean-of-ratios misreading the metric exists to avoid.
  It stays in the caption and in `analysis.json`.
* Canvas content: ticks, the two axis labels, one three-entry legend, the two
  reference lines. Controls (self/embed ≡ 0, other-game swaps ≪ 0, correct
  action unmoved) are caption / supplement material.

## 7. component-patching results and conclusions (full n = 165 states / 64 games)

| block | all_mix | all_mlp | MLP R_D / R_C (nats) | rand_all_mix |
|---|---|---|---|---|
| Qwen text | +100.0 [100.0,100.0] | **−34.7 [−41.7,−28.1]** | −2.34 / −0.000 | −286 |
| Qwen visual | +100.0 [100.0,100.0] | **−5.9 [−11.7,−0.6]** | −0.29 / −0.041 | −384 |
| Gemma text | +100.0 [100.0,100.0] | **+57.5 [+54.3,+60.7]** | +12.32 / +0.099 | −148 |
| Gemma visual | +100.0 [100.0,100.0] | **+56.2 [+51.0,+61.3]** | +5.73 / −0.030 | −287 |

All gates exact on every block (embed ≡ 0, self ≡ 0, ceiling ≡ 100.0 %,
reference-scorer difference 0.0 nats, 0 cells dropped; 1 same-game random donor
per block excluded by provenance).

* The MLP's causal contribution is **PEB-specific** everywhere (|R_C| ≤ 0.1 nats
  against |R_D| = 0.3–12.3 nats; weakest block Qwen visual at 14 % of R_D —
  footnote it).
* The MLP's **sign is architecture-dependent and consistent across modalities**:
  suppression on Qwen (−35 % text, −6 % visual), carriage on Gemma (+57 % / +56 %).
  CIs do not overlap. This falsifies the planned universal claim
  "token-mixing carries, MLP suppresses, in both models".
* Random-state donors never recover (−148 % to −384 %): the recovery is
  exposure-information, not arousal.
* **Conclusion licensed:** the same behavioural bias reaches the action
  decision in both architectures, distractor-side only, but through
  **architecture-contingent sublayer routing**; combined with layerwise-exposure-slope, the
  evidence becomes readable progressively with relative depth in both, while
  *which component causally carries it* is not shared.

## 8. Paper-ready Methods paragraphs (English)

**Exposure-depth profiling (`layerwise-exposure-slope/`).** For each of the 165 decision states, each
exposure level `k ∈ {0,1,2,4,8}`, each input modality and each of the two forced
actions, we ran a single forward pass and captured the pre-final-norm residual
stream after every decoder layer with forward hooks. We decoded the forced
action at layer `l` with the model's own readout — final RMSNorm followed by the
tied language-modelling head, including `final_logit_softcapping` where the
configuration defines it — and summed token log-probabilities over the action
span defined by stem-only tokenisation. Per state and layer we regressed the
decoded action margin `M_l(k) = s_l(a_D;k) − s_l(a_C;k)` on `k` by ordinary
least squares, yielding a layer-wise exposure slope `β_l`; the distractor- and
correct-side slopes were fitted separately and stored alongside. Uncertainty is
a game-clustered bootstrap (4000 resamples, seed 0). Two estimators are
reported and never mixed: the mean of per-state slopes, and a pooled regression
on across-state level means. Each model's top-layer slope was required to
reproduce the behavioural probe's published anchor for the same states and
k-set before any figure was drawn.

**Cross-architecture component patching (`component-patching/`).** For each state and modality
we ran three forwards: `k=8` (donor), `k=0` (host) and `k=0` again with the
output of a chosen component replaced, at the rows whose logits score the
candidate action, by the donor's output at the same action-token index. The two
components are the only sublayers present in every layer of both models: the
token mixer (Gated DeltaNet or softmax attention on Qwen3.5-9B; sliding or full
attention on Gemma-4-31B-it) and the MLP; both are patched at their module
output, before the residual addition. No recurrent state, convolution state or
key–value cache is touched. We report the recovered exposure effect,
`100·mean(R)/mean(G)` with `G = S_{k=8}(a_D) − S_{k=0}(a_D)` and
`R = S_patch(a_D) − S_{k=0}(a_D)`, as a ratio of means with a game-clustered
bootstrap. Controls: a self-patch (exact zero), the token embedding (exact
zero), the final residual (exact 100 %), and a random-state donor drawn from a
different game under a single shared donor map. Because both models are
row-wise outside their mixer modules, patching every mixer output at the
decision rows reproduces the donor score exactly; this ceiling is drawn as a
reference line and is not interpreted as evidence.

## 9. Caption caveats that must survive editing

1. Absolute layerwise-exposure-slope slopes are not comparable across models: `β` differs by
   ~3.3× because of Gemma's `final_logit_softcapping = 30` and the 280
   soft-token visual budget, not because the effect is stronger.
2. `figure_h1_*_norm` is per-action-token normalised; `_sum` is not. The two are
   different quantities and must not be mixed in one sentence; the colour scale
   is capped at the shared 98th percentile and clips above it.
3. The correct-action side is omitted from the h1 heatmaps (`β^C ≈ 0` on the
   shared scale) but is *not* equally flat in both models: max |β^C| is 6.4 % of
   the top-layer margin slope on Qwen and 14.5 % on Gemma.
4. In Fig. 5, the token-mixer cloud sits on the structural ceiling by
   construction (README §5.1); the informative clouds are the MLP swaps.
5. Gemma visual = `max_soft_tokens = 280`; Qwen visual = native raster. Vision
   token counts differ by an order of magnitude (≈1.3k vs ≈15k), which is part
   of why per-forward cost and absolute scales differ.
