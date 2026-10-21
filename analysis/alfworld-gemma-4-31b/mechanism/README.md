# Mechanism experiments on Gemma-4-31B-it

Gemma-4-31B-it counterparts of the Qwen3.5-9B mechanism experiments in
`../../alfworld-qwen3.5-9b/mechanism/` (see its `README.md` and `METHODS.md`).
They use the same decision states and k-controlled documents from the Qwen
tree; `lib/config.py` points `ANALYSIS_ROOT` at
`../../alfworld-qwen3.5-9b` and `lib/__init__.py` adds the Qwen `lib/` to
the import path.

| experiment | paper | notes |
|---|---|---|
| `symbol-encoding` | Fig. 3b | adds a tokenizer gate (`tok_check.py`) and a Gemma visual scorer (`symlib_gemma.py`) |
| `component-patching` | Fig. 5 | Gemma runner + `archlib_gemma.py`; shared spec / harness / analysis are symlinks into the Qwen component-patching |

```bash
cd symbol-encoding                && bash launch.sh
cd ../component-patching && bash launch.sh
```

symbol-encoding writes to `symbol-encoding/results/`, component-patching to `results/component-patching/gemma/`.
The component-patching analysis runs once from the Qwen component-patching directory and reads both
models' results.
