# Symbol Encoding — text symbol vs. re-recognised pixels (paper Fig. 3b)

## The question

Same skill content, two modalities, different exposure effect. Repeating a
distractor keyword k times raises logP(distractor action) with slope

| modality | slope d logP(distractor)/dk | source |
|---|---|---|
| text   | 0.741 (full k grid) / **0.846** on k∈{0,4,8} | LI, stem-only scoring |
| visual | **0.552** | LI |

Both modalities carry the *same characters, same words, same layout*. So the gap
has to come from what the keyword *becomes* on the language side. The hypothesis
to be tested: in text the keyword is a **native discrete token** the model was
trained to associate with the action schema; in visual it is **pixels that must be
re-recognised** through vision tower → 2×2 patch merge → projector, and that
channel is lossy.

"Native discrete token" is actually two claims that had to be separated:

* **COUNT / crispness** — the keyword is *one short* token, so exposure adds k
  clean symbol occurrences.
* **IDENTITY** — the keyword in the document is *the very same token* the model
  must emit in the action, so exposure is symbol→symbol reinforcement.

## Manipulation: casing

Casing is the cheapest manipulation that moves BPE segmentation while holding
everything else fixed:

| arm | document | keyword BPE (cabinet / microwave) | legible? | chars | k |
|---|---|---|---|---|---|
| `native` | `open cabinet 4` | 1 / 1 | yes | same | same |
| `title` | `open Cabinet 4` | 1 / 1 | yes | same | same |
| `upper` | `open CABINET 4` | 2 / 3 | yes | same | same |
| `altcase` | `open CaBiNeT 4` | 4 / 6 | yes | same | same |
| `ctrl_altcase` | keyword native, a **frequency-matched other word** mangled | 1 / 1 | yes | same | same |

Held constant: characters, word, semantics, human legibility, word count, slot
positions, k, and **the scored action string** (always the original lowercase
`open cabinet 4`). Only segmentation/surface moves.

`title` is the key control: it changes the surface *without* changing the token
count for most keywords, so it separates COUNT from IDENTITY.

**Rendered to pixels the identical strings change segmentation not at all** —
which is what makes the crossed design diagnostic. The signature is a
**modality × casing interaction**, not a main effect.

## Experiments

| id | script | what it does | GPU |
|---|---|---|---|
| **F0** | `reanalyze_li_by_tokenization.py` | split the existing 165 LI states by native keyword tokenization (1-token vs 3-token). Free. **Confounded with task family — direction only.** | none |
| **F1** | `run_text_casing.py` | text, 5 arms × k∈{0,4,8} × 165 states | 1 |
| **F2a** | `render_cased_cards.py` | re-render the skill cards with recased keywords (CPU, multiprocess) | none |
| **F2b** | `run_visual_casing.py` | visual, 3 arms × k∈{0,4,8} × 165 states | 1 |
| **F3** | `run_case_matched_action.py` | **rescue**: recase the *scored action* to match the document, restoring symbol identity while leaving the mangled form in place | 1 |
| **F4** | `run_symbol_readout.py` | modality-fair "which word appears most often" readout margin, logP(keyword)−logP(control), at a common text position | 1 |

Analysis: `analyze_casing.py` (F1+F2b, incl. the interaction),
`analyze_case_matched.py` (F3 rescue fraction).

## Predictions and how each outcome reads

| outcome | conclusion |
|---|---|
| text slope falls with casing, visual flat ⇒ **positive interaction** | text rides on the keyword being a native discrete symbol; the modality gap is an encoding difference. Hypothesis supported. |
| both fall equally (interaction ≈ 0) | surface-form salience, not tokenization. |
| neither falls | symbol crispness is not the carrier; look elsewhere. |
| `title` falls as much as `altcase`, token-count regression flat | not COUNT — test IDENTITY with F3. |
| F3 rescue ≈ 1 | the carrier is symbol **identity** (document token == emitted token). |
| F3 rescue ≈ 0 | the carrier is the mangled form itself (crispness/segmentation). |

### The prediction that actually mattered: STEP vs GRADED

The list above turned out to be too coarse, because **both** modalities lose slope
under casing (see Findings). The informative contrast is not *whether* casing
costs but **how the cost scales with the degree of mangling**, measured within
each modality and then differenced — which also cancels the large baseline level
gap between text (0.83) and visual (0.48):

* `upper` and `altcase` both break exact symbol identity, but `altcase` is far
  more visually irregular **and** carries ~2× the BPE tokens.
* **Discrete-symbol matching predicts SATURATION**: once the document token is no
  longer the token to be emitted, the match has already failed, so
  `altcase − upper ≈ 0`. A step function.
* **Perceptual re-recognition predicts GRADED decay**: recognition degrades
  continuously with irregularity, so `altcase − upper < 0`.
* The diagnostic is the **dose interaction**
  `(text: altcase − upper) − (visual: altcase − upper)`, positive iff text
  saturates while visual keeps degrading.

This is a stronger test than the original flat-vs-falling prediction, because it
does not require either modality to be insensitive to casing — only that they be
sensitive in *qualitatively different ways*.

## Method invariants (enforced in code, not just intended)

* **Scored span**: `action_start = len(tokenize(render(messages + [assistant: FORCED_STEM])))`
  — the LI-style stem-only tokenization (character offsets are off by one).
  Every cell records `stem_prefix_ok` (stem ids must be a literal prefix of the
  full forced sequence). logP is summed over **all** action tokens.
* **Both actions kept**: `logp_distractor` *and* `logp_correct`, so a collapse of
  the correct action can never be read as a boost of the distractor. logP(correct)
  slope is reported as a control in every table and must stay ≈ 0.
* **k=0 is a shared anchor** in F1/F2b: the k=0 document has no keyword, so all
  arms legitimately reuse one k=0 cell. In **F3 each arm gets its OWN k=0 anchor**,
  because recasing the scored action changes its token sequence and hence its
  absolute logP — slopes are comparable across arms, absolute logP is not, and
  absolute logP is never compared.
* **Invariants checked per cell, skip rather than fudge**: keyword whole-word
  count must equal k; `n_replaced == k`; document character length unchanged by
  recasing. Violations are written to disk as `{"skipped": reason}`.
* **The regressor is the MEASURED token count**, not the nominal arm — `safe`
  stays 1 token even uppercased, `microwave` goes 1→3→6.
* **Effect sizes first**: per-state paired deltas with bootstrap CIs **and**
  game-clustered bootstrap CIs (165 states come from only 64 games), plus sign
  tests. p-values are secondary. No averaging of per-state "recovery %" — the
  F3 rescue fraction is formed from the bootstrap of the two paired differences,
  because a per-state ratio explodes when a state's denominator is near zero.
* **Same backend and precision throughout** (bf16, sdpa, single device); the
  `native` arm *is* the no-intervention run, and the GATE below checks it.
* **No cross-modal positional patching.** Nothing is swapped between text and
  visual; each modality is manipulated within itself and the two are compared as
  an interaction.
* **No teacher-forced leakage.** F4 scores one candidate word at a fixed assistant
  stem at prefill; nothing is generated, and the stem carries no information about
  which word the answer is.

### GATE

The `native` arm must reproduce the stem-only baseline on k∈{0,4,8}:
text **0.846**, visual **0.552**. `analyze_casing.py` prints PASS/CHECK and
whether the anchor lies inside the bootstrap CI. If the gate fails nothing
downstream is interpretable.

## Known caveats, recorded up front

* **Card re-render.** `--verify-native` found 56/60 symbol-encoding `native` cards
  byte-identical to the LI originals; the 4 differences were Unicode
  box-drawing glyphs (`├── │ └──`) inside one code fence of
  `card0_alfworld-inventory-management.png`, 1508 pixels in the left margin at
  x∈[61,134] — a **font-fallback difference between the environment that rendered
  the LI cards and this one**, keyword-independent. Mitigation: **all** arms
  including `native` are re-rendered here, so every symbol-encoding comparison is
  internally apples-to-apples, and the substantive gate is the LI *slope*, not
  byte identity.
* **F0 is confounded**: keyword tokenization tracks task family
  (`sinkbasin`↔clean, `desklamp`↔look-at-obj-in-light). Per-keyword slopes are
  printed so a reader can see whether any split is one keyword's doing.
* **`ctrl_altcase` is not a null-effect control by assumption** — mangling any
  word can perturb the document slightly. Its measured value is the noise floor
  the keyword arms are judged against.
* **Control words for the F1/F4 margin are frequency-matched but not
  concreteness-matched** (they are abstract meta-vocabulary).
* **At k=8 there is no FILLER control region**
  (`n_fil_tokens=0`: all 8 slots are injected). Any region-level follow-up must
  respect that.
* **GDN**: this experiment is deliberately behavioural. It makes no claim about
  which sublayer carries the effect, and in particular does not assume the 8
  softmax layers do.

## Results

Fig. 3b (keyword readout across exposure, Qwen3.5-9B and Gemma-4-31B-it) is
drawn by `paper_figures/plot_fig3b_symbol_readout.py` from the F4
`results/symbol_readout` outputs of both models.
