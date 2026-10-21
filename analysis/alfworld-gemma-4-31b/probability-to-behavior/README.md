# Probability-to-Behavior on Gemma-4-31B-it

Same probe as `../../alfworld-qwen3.5-9b/probability-to-behavior/` (see its
`README.md`), run on Gemma-4-31B-it over the **same** states and k-controlled
documents, which are read from the Qwen tree:

    ../../alfworld-qwen3.5-9b/probability-to-behavior/states/
    ../../alfworld-qwen3.5-9b/probability-to-behavior/variants/

Run the Qwen pipeline's offline stages first. This directory runs only
probe (text, rasterized, structural × k ∈ {0,1,2,4,8}) and analyze; no
rollouts.

```bash
bash run_full.sh          # starts vLLM (2 servers, TP=2), probe, analyze
SKIP_VLLM=1 bash run_full.sh
bash stop_vllm.sh
```

Outputs are written to `results/` and `results/analysis/`.
