# Symbol Encoding on Gemma-4-31B-it

Same experiment as `../../../alfworld-qwen3.5-9b/mechanism/symbol-encoding/`
(see its `README.md` for the method), run on Gemma-4-31B-it. The Python files
are the Qwen ones, with two Gemma-specific additions:

- `tok_check.py` — tokenizer gate run first. symbol-encoding is only meaningful if the
  tokenizer produces DIFFERENT token sequences for the casing arms
  (`cabinet` / `Cabinet` / `CABINET` / `CaBiNeT`, plus a frequency-matched
  control word). Qwen uses BPE, where casing changes segmentation; if Gemma's
  SentencePiece tokenizer normalised case, the manipulation would be a null
  intervention. The gate exits with code 2 if any arm collapses onto `native`.
- `symlib_gemma.py` — replaces only the visual scorer of `symlib.py`, which is
  Qwen-VL specific (`image_grid_thw`, `merge_size`). `run_visual_casing.py`
  and `run_symbol_readout.py` install it and assert that it is active.

## Inputs (read-only, from the Qwen tree)

    ../../../alfworld-qwen3.5-9b/lexical-interference/states/     decision states
    ../../../alfworld-qwen3.5-9b/lexical-interference/variants/   k = 0/1/2/4/8 documents
    ../lib/                                                       config (+ Qwen lib on sys.path)

Outputs are written to `results/` in this directory.

## Run

    # all stages: tok -> F0 -> F1 -> F2a -> F2b -> F3 -> F4 -> analysis
    bash launch.sh

    # selected stages
    STAGES="f4,analysis" bash launch.sh

`MECH_MODEL` defaults to `$MODELS_DIR/Gemma-4-31B-it`; shards are spread over
`MECH_GPUS` (default `0,1,2,3`, `NSHARDS=4`).
