# Component Patching on Gemma-4-31B-it

The experiment is defined in
`../../../alfworld-qwen3.5-9b/mechanism/component-patching/README.md`;
section references (`README §…`) in this directory's code point there.

- `run_component_patch.py` — Gemma runner (same spec, donor map, gates
  and metric as the Qwen runner).
- `archlib_gemma.py` — reads the layer plan from the loaded Gemma-4 config and
  provides the logit lens; `python3 archlib_gemma.py --selftest` checks it.
- `patch_spec.py`, `patch_harness.py`, `analyze_component_patch.py`, `selftest_*.py` — symlinks into
  the Qwen component-patching directory.

```bash
bash launch.sh                      # STAGE=headline (default) | bands | all
STAGE=all MODS="text visual" bash launch.sh
```

Results go to `../results/component-patching/gemma/soft<SOFT>/`. Then run the analysis from the Qwen
component-patching directory (`python analyze_component_patch.py`), which reads both models.
