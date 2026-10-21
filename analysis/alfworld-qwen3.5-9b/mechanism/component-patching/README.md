# Component Patching — architecture-agnostic, cross-model (paper Fig. 5)

One intervention, defined identically on two architecturally different models:
**Qwen3.5-9B** (32 layers; 24 Gated DeltaNet + 8 full attention) and
**Gemma-4-31B-it** (60 layers; 50 sliding attention + 10 full attention).

This file is the single source of truth. The Gemma tree does not carry a copy —
`patch_spec.py`, `patch_harness.py`, `analyze_component_patch.py` and the self-tests are **symlinks
into this directory**, so "same definition" is a byte-level fact, not an intention. Only
the two runners differ, and they differ solely in cell construction (tokenizer /
processor / model-loading protocol), never in what is patched or how it is
scored.

---

## 1. Question, and how it differs from layerwise-exposure-slope

| | question | evidence type |
|---|---|---|
| `layerwise-exposure-slope` | **Where does exposure-dependent action evidence become readable?** | observational / decodability (logit lens) |
| `component-patching` | **Which shared computation causally carries that evidence to the action decision?** | causal (activation patching) |

layerwise-exposure-slope's logit lens says a layer's residual stream *encodes* the exposure
difference. It cannot say the layer's computation *does* anything with it. component-patching
transplants the exposure difference into a low-exposure run and measures how
much of the behavioural gap reappears.

## 2. Why only two components

The two models share almost no architecture-specific objects. Qwen's Gated
DeltaNet recurrent/conv state has no Gemma counterpart; Gemma's sliding-window
KV cache has no Qwen counterpart. What **every** layer of **both** models has is
exactly one token-mixing sublayer and exactly one MLP sublayer. So component-patching is
defined on those two abstractions and on nothing else:

| | Qwen3.5-9B | Gemma-4-31B-it |
|---|---|---|
| layers | 32 | 60 |
| token mixer | `linear_attn` (GatedDeltaNet, 24×) or `self_attn` (8×) | `self_attn` (50× sliding + 10× full) |
| **Token Mixer** patch site | mixer module output | attention module output |
| **MLP** patch site | `mlp` module output | `mlp` module output |
| patch point | after the module completes, before the residual add | after the module completes, before the residual add |
| depth | `d = l/31` | `d = l/59` |

Never touched: GDN recurrent state, GDN conv state, KV cache, attention weights,
RoPE phases, the LM head.

### 2.1 One real architectural asymmetry, stated not hidden

Verified from the installed `transformers` sources (`Qwen3_5DecoderLayer.forward`,
`Gemma4TextDecoderLayer.forward`):

```
Qwen3.5   x + mixer(input_layernorm(x))                       ← module output is added raw
Gemma-4   x + post_attention_layernorm(self_attn(input_layernorm(x)))
```

Gemma-4 has **sandwich norms**: a `post_attention_layernorm` and a
`post_feedforward_layernorm` sit *between* the module output and the residual
add, and the layer output is additionally scaled by `layer_scalar`. Qwen3.5 has
neither. Patching "the module output" is therefore the same *definition* on both
models, but the patched vector's fate differs: on Qwen it enters the residual
stream with its magnitude intact, on Gemma it is RMS-normalised first, so only
its **direction** survives.

Consequence, and how it is bounded: if the exposure effect lives partly in
magnitude, Gemma's recovered fraction is attenuated relative to Qwen's by
construction. Two Gemma-only Appendix arms quantify this —
`all_mix_postnorm` / `all_mlp_postnorm` patch the tensor *actually added to the
residual* (post sandwich-norm). On Qwen those arms are not merely similar to
`all_mix` / `all_mlp`, they are **the same tensor**, which is why the headline
uses the module-output definition: it is the one that exists in both models.
The headline Figure is unaffected by this choice on the Qwen side.

Also verified for this checkpoint: `enable_moe_block = False` and
`hidden_size_per_layer_input = 0`, so `layer.mlp` is the *whole* MLP
contribution — no MoE expert branch bypasses the patch site. Had MoE been on,
"MLP" would have had to be defined as `mlp + experts` and this design would be
wrong.

## 3. Material — identical for both models

* States: the **165 LI decision states** (`lexical-interference/states/`,
  64 games) — the same population `layerwise-exposure-slope` used on both sides, so Figure (a) and
  Figure (b) describe the same states and `G` is comparable to layerwise-exposure-slope's anchors.
  The Gemma tree reaches them through `lib/config.py`, which hard-points
  `ANALYSIS_ROOT` at the Qwen tree; the documents are model-independent by
  construction and are never re-generated.
* Exposures: **k = 0 and k = 8 only** (`variants/{text,visual}/<sid>/level_{0,8}`).
* Modalities: Text and Visual. Gemma's visual arm runs at `max_soft_tokens=280`,
  the resolution used by the behavioural probes, so the gate is the LI visual
  anchor. Qwen's visual arm is the native LI card render.
  **Absolute log-prob scales are not comparable across models** (different
  vision token budgets, different softcap); only the *recovered fraction* is.
* Actions: `correct_action` (a_C) and `distractor_action` (a_D) exactly as in
  the PEB probe. Both are always scored; neither is ever reduced to a margin
  before storage.

## 4. Row alignment — why prompt length is not a problem

k=0 and k=8 prompts differ in length (different documents, and on the visual arm
a different number of image tokens). Nothing is aligned by absolute position.
The scored span is `[action_start, T)` with `action_start` from the **stem-only
tokenisation** (`FORCED_STEM = "Thought: I will now perform the action.\nAction:"`),
never from character offsets, which are off by one and cost 32% of the
measured slope on Qwen.

The decision rows are the rows whose logits predict the action tokens:
`rows = [action_start-1, T-1)`. Alignment across exposures is **by action-token
index**:

> host row predicting action token *s*  ←  donor row predicting action token *s*

Two per-cell assertions gate every state, and the analyzer drops any cell where
either is false:

* `stem_prefix_ok` — the stem is a literal token prefix of both forced turns.
* `action_ids_match` — the k=8 and k=0 action token id sequences are identical.
  This is what makes index alignment legitimate rather than convenient.

Only decision rows are recorded or replaced. Prefill rows are never touched: the
intervention is confined to the rows that decide the action.

## 5. Intervention

Single full forward, `use_cache=False`, no cache surgery of any kind.

1. **Donor pass** — forward k=8, recording every patch site's output at the
   decision rows. Yields `S_{k=8}` and the donor activations in one pass.
2. **Host pass** — forward k=0, recording the same sites. Yields `S_{k=0}` and
   the host's own activations (which is what makes the self-patch control exact
   rather than approximate).
3. **Patched pass** — re-run k=0, replacing the chosen sites' outputs at the
   decision rows with the donor's, then recompute the action score.

`delta_T = action_start(k=8) − action_start(k=0)` is recorded per cell. The
donor vector carries no position of its own and the host row keeps its own RoPE
phase, so this is second order — but it is written into every record and the
analyzer correlates the effect against it instead of assuming it away.

## 5.1 What `all_mix` can and cannot show — a structural ceiling

`all_mix` (every layer's mixer output patched at the decision rows) will come out
at **≈ 100% recovery on both models, by construction**, and it is important not
to read that as a finding about exposure.

Both architectures are row-wise everywhere *except inside the mixer module*: the
residual add, every RMSNorm (including Gemma's sandwich norms) and the MLP
(Linear → activation → Linear) all act per row. So at a decision row *r*, if

1. the embedding at *r* is identical between donor and host — guaranteed by
   `action_ids_match`, the same assertion that licenses index alignment, and
   independently verified by the `embed` arm coming out exactly zero; and
2. every mixer output at *r* is replaced by the donor's,

then by induction the residual at *r* matches the donor's after every layer, and
the patched score equals `S_{k=8}` exactly. Overwriting the mixer output *after*
the cross-row mixing has already happened is precisely what makes this exact: the
unpatched document rows can no longer reach row *r*. `all_mix` therefore lands on
the same ceiling as `resid_last`, and the toy-model harness test asserts the two
are equal to 1e-6 (`selftest_patch_harness.py`, §3).

What survives as evidence:

* **`all_mlp` is not pinned.** Patching only the MLPs leaves every mixer at its
  host value, so nothing forces it to the ceiling.
* **Subset arms localise.** A single depth band cannot reach the ceiling, because
  the layers outside it stay at host values. Band arms are therefore the
  quantitative depth result, and they are paired two ways:
  * `mix_band{b}` — **sufficiency**: can this relative-depth range carry the
    effect on its own?
  * `mix_all_but_band{b}` — **necessity**: is this range required? Dropping the
    range that writes the evidence should cost the most recovery.
* **The correct-action arm** still discriminates: a patch that lifts a_C as much
  as a_D is arousal, not PEB, whatever its recovery of G.
* **The random-state control** still discriminates: another game's k=8 mixer
  outputs must not reproduce the effect.

So the cross-model headline is the **contrast** (Token Mixer at ceiling vs MLP
below zero, in both architectures) plus the **band profile**, not the absolute
height of the mixer bar. Figure (b) must draw the ceiling explicitly — the
`resid_last` anchor is plotted as a reference marker — so a reader can see that
the mixer bar is bounded by construction and the MLP bar is not.

## 6. Metric — Recovered exposure effect

Per state, per arm:

```
G = S_{k=8}(a_D) − S_{k=0}(a_D)              exposure gap
R = S_patch(a_D) − S_{k=0}(a_D)              recovered by the patch
Recovered exposure effect = 100 × R / G
```

Reported as **mean(R) / mean(G)**, never as the mean of per-state ratios: some
states have a near-zero `G` and a per-state ratio diverges there. The point
estimate and the CI are both ratio-of-means.

Uncertainty: **game-clustered bootstrap, 4000 resamples, seed 0** — resample the
64 games, average within game, then across games, and recompute the ratio inside
each bootstrap sample. 165 states come from only 64 games, so a state-level
bootstrap would understate the CI. Same estimator convention as layerwise-exposure-slope and LI.

Because the metric is a ratio, Qwen and Gemma are comparable despite log-prob
scales that differ by a large constant factor.

## 7. Controls — exactly three

| control | arm | required result |
|---|---|---|
| **Self patch** | `self_all` (donor = the host's own k=0 activations, every patched site) | ≈ 0 exactly. Anything else means the harness does not reproduce its own forward pass. |
| **Random-state patch** | `rand_all_mix`, `rand_all_mlp` (donor = a different state's k=8 activations, from a **different game**) | must not stably recover the exposure effect. |
| **Correct-action** | every arm also reports `delta_correct` / `recovered_correct` | if a patch lifts a_D and a_C together it is arousal, not PEB. |

Plus two scale anchors, both of which must hold exactly:

* `embed` — token embedding at the decision rows. The tokens are identical by
  construction, so this must be **exactly 0** (`moved == 0` as well as `delta == 0`).
* `resid_last` — the final decoder layer's output at the decision rows. This
  determines the logits, so donating it must reproduce the donor's score:
  **recovery = 1.000**. Verified analytically for both models: nothing sits
  between `layers[-1]`'s output and the final norm.

The random donor is chosen once, deterministically (seed 0), game-disjoint, and
the map is written to `alfworld-qwen3.5-9b/mechanism/results/component-patching/donor_map.json` and shared by both trees so the
two models are given the *same* mismatched donors. Each state's donor is the
previous state in the shuffled order, held in a rolling buffer; the first state
of each shard therefore has no random arm (`n_rand = n_states − n_shards`).
Where the donor's action has a different token count, only the overlapping
prefix is patched and `rand_n_patched` / `rand_n_host` are recorded.

## 8. Arms

**Stage 1 — headline (Figure b).** Both models, identical names:

`self_all`, `embed`, **`all_mix`** (Token Mixer, every layer), **`all_mlp`**
(MLP, every layer), `resid_last`, `rand_all_mix`, `rand_all_mlp`.

**Stage 2 — depth localisation (Appendix).** Bands are fixed *a priori* on
`d = l/(L−1)` with edges `[0, .25, .50, .75, 1]` and are **identical for both
models**; no layer is ever selected after seeing a result.

* sufficiency: `mix_band0..3`, `mlp_band0..3`
* necessity: `mix_all_but_band0..3`, `mlp_all_but_band0..3`

For L=32 the bands are layers 0–7 / 8–15 / 16–23 / 24–31; for L=60, 0–14 /
15–29 / 30–44 / 45–59. The bands contain different numbers of layers because the
models have different depths — that is the point of normalising depth, and it is
why absolute layer numbers never appear in the main text. The sufficiency and
necessity arms partition the same layer set (`band ∪ all-but-band = all`,
`band ∩ all-but-band = ∅`, asserted for both components on both models in
`selftest_patch_spec.py`), so the two rows of the Appendix figure are comparable.
Per §5.1 the band arms, not `all_mix`, are what localises the effect.

**Arch-specific (Appendix, never in a cross-model comparison):**

* Qwen: `all_mix_full` (8 softmax layers), `all_mix_lin` (24 GDN layers).
* Gemma: `all_mix_postnorm`, `all_mlp_postnorm` (§2.1 magnitude bound).

## 9. Gates (all must pass before any figure is drawn)

1. `embed`: `delta == 0` and `moved == 0`, exactly, every cell.
2. `self_all`: `|delta| < 1e-6` on both actions, every cell.
3. `resid_last`: `|recovery − 1| < 1e-3`, every cell.
4. `stem_prefix_ok` and `action_ids_match` true on every retained cell.
5. Reference consistency: the runner's own `S_{k=0}` must match the probe-family
   scorer (`symlib.score_actions_*`) on the first cell of each modality within
   the declared tolerance. Single-forward on both sides here, so the tolerance
   is bf16-level.

## 10. Outputs

```
# paths relative to analysis/alfworld-<model>/mechanism/
results/component-patching/qwen/per_cell_<stage>/<modality>/*.json           Qwen cells, one file per cell, resume-safe
results/component-patching/gemma/soft280/per_cell_<stage>/<modality>/*.json  Gemma cells (Gemma tree)
# written by analyze_component_patch.py, in the Qwen tree only
results/component-patching/analysis.json                                     analyzer output (both models)
results/component-patching/summary.csv                                       per-arm summary
results/component-patching/per_state.csv                                     per-state G / R (Fig. 5)
results/component-patching/donor_map.json                                    shared by both trees, generated once
```

Fig. 5 is drawn by `paper_figures/plot_fig5_component_patching.py` from `per_state.csv` / `analysis.json` written by `analyze_component_patch.py`.

## 11. Claim boundaries

* component-patching licenses: *this component, at these rows, causally carries
  exposure-dependent evidence to the action decision, in both models.*
* component-patching does **not** license: which head, which circuit, or how the evidence was
  written into the mixer's input in the first place.
* Cross-model comparability covers the **recovered fraction and its sign**, not
  absolute nats: vision token budget, softcap (Gemma 30.0, Qwen none) and
  sandwich norms all differ. The Gemma caption must carry the
  `max_soft_tokens=280` note, as F3 does.
