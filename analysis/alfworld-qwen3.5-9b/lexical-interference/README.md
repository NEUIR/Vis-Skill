# Lexical Interference — keyword frequency → probability shift → final error

Controlled single-step probe of the two main experiments
(`experiments/alfworld/qwen3.5-9b/{text-skill,vis-skill}`, Qwen3.5-9B on ALFWorld).

**Question.** When a *distractor* action keyword (e.g. `microwave` in
`open microwave 1`) occurs `k` times in the Skill document but is **not
applicable in the current state**, does the keyword frequency shift the
model's next-action distribution — and does that shift differ between the
**Text Skill** and the **Visual Skill** (same document, same content, only
the modality differs)?

**Hypothesis under test.** Text skills show a steep
frequency → probability → error chain; visualisation flattens it
(visualisation weakens the direct effect of Skill keyword frequency on
subsequent decision probabilities, i.e. weaker lexical dependency).

Design summary:

| | lexical-interference |
|---|---|
| states | **replayed in the real ALFWorld env** → true `admissible_commands` |
| distractor | verified **not admissible now**, keyword **in the skill corpus**, **not in query / correct action** |
| manipulation | keyword scrubbed from base docs; k injections vs 8−k word-count-matched fillers at fixed slots → **exact** keyword count = k, **exactly constant** total word count (asserted) |
| forced decoding | **every candidate action** → full normalised distribution, entropy |
| logprob comparability | **constant stem excluded** (`Thought: ...\nAction:` scored separately) → comparable across k and modality |
| analysis | P/logP/entropy trends, **Skill-word Reproduction Rate**, **bad-case quadrants** (Text wrong / Visual right …), paired per-state k0→k8 deltas with sign tests |

## The chain, operationalised

1. **Extract decision states** (`code/extract_decision_states.py`)
   Replays successful trajectories (default: games solved by *both* main runs,
   trajectory taken from the text run) inside ALFWorld to recover, at each
   step t: clean history (system + initial obs + ReAct turns, **without** any
   procedural-guidance injections), `correct_action`, and the
   `admissible_commands` candidate set. ≤3 states per trajectory, first and
   last step dropped, failed steps dropped.

2. **Choose the distractor action / keyword** (`code/skill_lexicon.py`)
   A distractor must satisfy all of:
   (a) plausible for the task family & present in the room
       (priority: premature `open microwave/fridge`, premature `heat/cool/clean`, …);
   (b) **NOT** in the current `admissible_commands` (verified by env replay);
   (c) different verb than the correct action;
   (d) its keyword occurs in the 37-document skill corpus but **not** in the
       task query and **not** in the correct action → any later reproduction
       of that word is attributable to the Skill document.
   Canonical example: correct = `take apple 1 from countertop 1`,
   distractor = `open microwave 1`, keyword = `microwave`.

3. **Control keyword frequency; Text / Visual inputs** (`code/construct_variants.py`, `code/render_cards.py`)
   Base document = the state's top-5 skills with every whole-word occurrence
   of the keyword replaced 1:1 by a neutral word (`microwave`→`appliance`),
   so word count is untouched and k=0 truly has zero mentions. 8 fixed,
   evenly-spaced insertion slots; in condition k, k slots carry an injection
   sentence naming the distractor **exactly once**, the other 8−k carry the
   paired filler sentence — same slot, same register, **same word count**
   (asserted per state). `k = 0, 1, 2, 4, 8`. Injections are *spread* evenly
   over slots so k does not covary with position.
   Visual = the identical document rendered faithfully into one PNG card per
   skill (v5 style, 1000 px, DejaVu; `--verify` reports any drawn-vs-source
   character loss > 2 %).

4. **Action probability distribution, not only the final choice** (`code/probe_lib.py`, `code/run_probe.py`)
   Per (state × k × {text, visual}) cell:
   * free generation, temperature 0 → the action actually taken;
   * forced decoding of **every candidate action** under a byte-identical
     prefix (`continue_final_message` + `prompt_logprobs`, the only path that
     keeps image context intact);
   * the constant stem `Thought: I will now perform the action.\nAction:` is
     tokenised once and **subtracted**, so `logp_*_action` is comparable
     across k and across modality.
   Reported per cell:
   `P(correct)`, `P(distractor)` (softmax over the candidate set; whole-
   sequence `_sum` and length-normalised `_mean`), raw
   `logP(correct)`, `logP(distractor)` (stem-excluded), and
   `H = −Σᵢ pᵢ log pᵢ` (nats + normalised by log|candidates|).
   A **no-skill control** cell (no document at all) is probed per state as an
   anchor.

5. **Is the word reproduced?** (`code/analyze_results.py`)
   Skill-word Reproduction Rate =
   `#(parsable errors whose action contains the distractor keyword) / #(all parsable errors)`,
   per (k × representation), plus stricter/looser variants:
   * `repro_exact_rate` — error action **equals** the distractor action;
   * `repro_phrase_rate` — error action contains the verb phrase
     (`open microwave`);
   * `repro_other_skillword_rate` — error action contains a **document word
     that occurs nowhere in query/history/correct action** (other-skill-word
     penetration beyond the injected keyword).

6. **Bad-case analysis** (`code/analyze_results.py`)
   Pair text and visual cells of the same (state, k) into quadrants:
   `both_correct` / `text_wrong_visual_right` / `text_right_visual_wrong` /
   `both_wrong`; exact-binomial sign test on discordant pairs. Inside each
   discordant quadrant: the wrong side's keyword-reproduction rate. If
   Text-wrong cells reproduce the high-frequency skill word far more than
   Visual-wrong cells, the story holds.

The paper figures (Fig. 1a, 1b) are drawn by the scripts in
`paper_figures/` from `results/analysis/per_trial.csv`.

## Layout

```
lexical-interference/
├── launch.sh               # staged pipeline (STAGES/SMOKE/LIMIT/MAX_GAMES knobs)
├── start_vllm.sh           # 4×Qwen3.5-9B, TP=1, ports 8000-8003, image:6
├── stop_vllm.sh
├── code/
│   ├── skill_lexicon.py    # grammar normalisation, corpus stats, distractor
│   │                       #   choice, keyword scrubbing, slot texts (k payload)
│   ├── prompts.py          # system prompt (byte-identical to main runs) + wrappers
│   ├── extract_decision_states.py   # step 1 (env replay)
│   ├── construct_variants.py        # step 2 (k documents + assertions)
│   ├── render_cards.py              # step 2b (faithful PNG cards + verify)
│   ├── probe_lib.py        # prompt assembly, free gen, forced decoding, stats
│   ├── run_probe.py        # step 3 (dynamic 128-worker scheduling, resume-safe)
│   ├── smoke_test.py       # single-state end-to-end verification
│   ├── analyze_results.py  # step 4 (chain analysis → JSON/CSV)
│   └── base_config.yaml    # ALFWorld env config (valid_unseen replay)
├── paper_figures/          # Fig. 1a / 1b plotting scripts (read results/analysis/per_trial.csv)
├── states/                 # (generated) decision states + manifest.json
├── variants/               # (generated) text/{state}/level_k.md(+parts), visual/{state}/level_k/card*.png
└── results/                # (generated) per-cell JSONs; analysis/ (json, csvs)
```

## Run

```bash
cd analysis/alfworld-qwen3.5-9b/lexical-interference

# 1) offline stages (need ALFWORLD_DATA at data/alfworld, no GPUs)
STAGES="extract,construct,render" bash launch.sh

# 2) verify the chain on one state before the big run (needs servers)
bash start_vllm.sh
SMOKE=1 STAGES="" bash launch.sh          # or: python code/smoke_test.py --states_dir states --variants_dir variants

# 3) probe + analyze
STAGES="probe,analyze" bash launch.sh
# or everything in one go:
bash launch.sh

bash stop_vllm.sh
```

Useful knobs: `LIMIT=20` (probe only 20 cells), `MAX_GAMES=10` (replay 10
games), `LEVELS=0,8` (quick coarse sweep), `STAGES=analyze` (re-run
analysis only). All stages are resume-safe: existing outputs are skipped
(probe) or overwritten deterministically (construct/render/analyze).

## Cost

Measured inputs: text run solved 85/129 games, visual run
91/131, **both-solved = 62 games** → ≤3 states each ≈ **150–190 states**;
per state 5 levels × 2 representations + 1 no-skill control = 11 cells →
**≈1700–2100 probe cells**. Per cell: 1 free generation + 2 cheap tokenize
calls + 2 HTTP calls per candidate (≤16 candidates → ≤17 GPU-heavy calls).

Wall-clock estimate on 4×A100-80GB (≈30 GPU-s per visual cell):

| stage | time |
|---|---|
| extract (replays ≤~134 env games, resets included) | ~10–20 min |
| construct | ~1 min |
| render (~150–190 states × 5 levels × ≤5 cards ≈ 4–5k PNGs, serial) | ~30–60 min |
| probe — text cells (~800, 17 heavy text prefills each) | ~3–4 h |
| probe — visual cells (~800, 17 heavy 5-card prefills each, no prefix cache) | ~10–14 h |
| probe — control cells (~170) | ~20–30 min |
| analyze | ~2–5 min |
| **total** | **≈ 15–20 h** (probe dominates; visual half is the bottleneck) |

With vLLM V1, prefix caching may be auto-disabled for Qwen3.5 (a hybrid
GDN/linear-attention model). If prefix caching can be enabled
(`PREFIX_CACHING=1 bash start_vllm.sh`, then confirm
`enable_prefix_caching=True` in vllm_8000.log), the 16 per-candidate
prompt_logprobs calls share one cached prefix and the probe drops to
**≈ 3–4 h total**. If the server refuses to start with the flag, just rerun
without it.

Shortening levers (all design-visible, choose consciously):
* `LEVELS="0,8" bash launch.sh` — coarse sweep, ≈ halves probe time;
  the trend claim survives with 2 points + the paired Δ(k0→k8) test, but you
  lose the shape of the curve.
* `--max_candidates 8` at the extract stage — ≈ halves the forced-decoding
  calls (probe ≈ 7–9 h); entropy is then over ≤8 verbs-balanced candidates
  instead of ≤16.
* probe text first (`run_probe.py --representations text`, ≈ 3–4 h), analyze,
  then visual — the text-side curve and reproduction rates are reportable
  before the visual half finishes (all stages are resume-safe).
* calibrate before committing: run `smoke_test.py` (times one text + one
  visual cell serially) and multiply: total ≈ n_cells × t_cell / (4 GPUs ×
  ~2–3 batching speedup).

## Caveats / design notes

* **`put`→`move` normalisation.** This ALFWorld build accepts
  `move {obj} to {recep}` and rejects `put {obj} in {recep}`; everything is
  canonicalised to the `move … to …` surface form so free generations,
  admissible commands and forced-decoding targets are comparable.
* **Both-solved pool.** States come from games solved by *both* main runs →
  no per-modality selection bias. (`--require text|visual|any` available.)
* **Clean history.** The procedural-guidance messages the main runs injected
  are dropped from the replayed context; the only skill text in a probe
  prompt is the controlled document. History is identical across
  representations and across k.
* **"Correct" = the successful trajectory's action.** Other actions may also
  be environment-valid. `argmax`/rank of the forced distribution and
  `chosen_in_candidates` are logged for sensitivity checks.
* **Unparsable generations** count as errors in `error_rate` but are excluded
  from Reproduction-Rate denominators (`n_unparsable` is reported).
* **QA gates.** `construct_variants.py` asserts keyword count == k and
  constant word count per state; `analyze_results.py` re-measures both from
  the documents and reports cells with incomplete forced decoding
  (`n_scored < n_candidates`) or stem-tokenisation mismatch — those cells'
  distribution stats are computed only over fully-scored candidates.
