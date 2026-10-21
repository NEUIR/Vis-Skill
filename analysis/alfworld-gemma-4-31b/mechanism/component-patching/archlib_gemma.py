#!/usr/bin/env python3
"""
Architecture introspection + logit lens for Gemma-4.

WHY THIS FILE EXISTS
--------------------
The Qwen layerwise-exposure-slope hard-codes its backbone into three places:

    NUM_LAYERS      = 32
    FULL_ATTN_LAYERS= [3, 7, 11, 15, 19, 23, 27, 31]   # 3x GDN -> 1x full, x8
    hs[l + 1]                                          # "output of layer l"

None of the three survives the move to Gemma-4-31B, which is a 5:1
sliding/full hybrid with a different depth and a *different hidden-state
protocol*.  Everything architecture-dependent is therefore derived here, at
run time, from the loaded config and module tree.  Nothing about depth or
layer type is written down as a constant.

THE THREE GEMMA-4 FACTS THAT CHANGE THE LENS
--------------------------------------------
1. `layer_types` is explicit in the config.
   `Gemma4TextConfig.__post_init__` builds
       ["sliding_attention" if (i+1) % 6 else "full_attention" ...]
   and then *forces* the last layer to `full_attention`.  For a 60-layer
   checkpoint that yields full attention at L5,11,17,23,29,35,41,47,53,59 —
   10 full + 50 sliding.  We read the list
   rather than recomputing the pattern, because a checkpoint is free to ship
   an arbitrary one.

2. `final_logit_softcapping` may be non-None.
   `Gemma4ForConditionalGeneration.forward` applies
       logits = tanh(logits / cap) * cap
   after the LM head.  A lens that skips this is NOT the model's own readout,
   so the layer-31-equals-LI gate would fail for a reason that has nothing to
   do with the experiment.  `lens_logits()` replicates it.

3. `output_hidden_states=True` does NOT give you the last layer's raw output.
   In transformers 5.x the hidden states are collected by forward hooks
   (`utils/output_capturing.py`), and `capture_outputs(tie_last_hidden_states
   =True)` — the default — *overwrites* `hidden_states[-1]` with
   `last_hidden_state`, which for `Gemma4TextModel` is already
   post-`self.norm`.  Applying the final RMSNorm to that row again double-
   norms it.  Gemma's RMSNorm is `x/RMS(x) * weight` with a learned weight, so
   the second application is not idempotent and the top-layer lens would be
   quietly wrong — exactly the class of silent error the layerwise-exposure-slope gate exists to
   catch.

   `LayerTap` therefore installs its own forward hooks on the decoder layers
   and keeps each layer's *raw* output.  `h[l]` is unambiguously the
   pre-final-norm residual stream after layer l, for every l including the
   last.  As a bonus the hook slices the ~12 rows we actually score before
   storing anything, so peak memory is ~61 x 12 x H instead of the Qwen
   version's 33 x T x H (4.4 GB at T~16.5k).

CPU-ONLY SELF TEST
------------------
`python3 archlib_gemma.py --selftest` verifies the config-derived layer plan
and the depth-fraction mapping against a synthetic config, with no weights and
no GPU.  Use it to check this module before queueing a GPU run.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# config introspection
# ---------------------------------------------------------------------------
FULL = "full_attention"
SLIDING = "sliding_attention"


def text_config(config):
    """The text sub-config of a multimodal Gemma-4 config (or the config
    itself, if it is already a text config)."""
    getter = getattr(config, "get_text_config", None)
    if callable(getter):
        try:
            tc = getter()
            if tc is not None:
                return tc
        except Exception:
            pass
    return getattr(config, "text_config", config)


def layer_plan(config) -> dict:
    """Derive the depth and the per-layer mixer type from the config.

    Returns a dict that is written verbatim into every result file, so a
    figure can always be traced back to the architecture it was drawn from.
    """
    tc = text_config(config)
    n = getattr(tc, "num_hidden_layers", None)
    if not isinstance(n, int) or n <= 0:
        raise RuntimeError(f"could not read num_hidden_layers from {type(tc).__name__}")

    types = list(getattr(tc, "layer_types", None) or [])
    if len(types) != n:
        # No explicit list (or a stale one): fall back to Gemma-4's documented
        # 5:1 default, last layer forced to full attention.  Recorded as
        # `derived` so the provenance is never ambiguous.
        period = 6
        types = [SLIDING if bool((i + 1) % period) else FULL for i in range(n)]
        types[-1] = FULL
        source = f"derived(period={period}, last forced {FULL})"
    else:
        source = "config.layer_types"

    full = [i for i, t in enumerate(types) if t == FULL]
    sliding = [i for i, t in enumerate(types) if t != FULL]
    return {
        "num_layers": n,
        "layer_types": types,
        "layer_types_source": source,
        "full_attn_layers": full,
        "sliding_layers": sliding,
        "n_full_attn": len(full),
        "n_sliding": len(sliding),
        "sliding_window": getattr(tc, "sliding_window", None),
        "hidden_size": getattr(tc, "hidden_size", None),
        "final_logit_softcapping": getattr(tc, "final_logit_softcapping", None),
        "tie_word_embeddings": getattr(tc, "tie_word_embeddings", None),
        "num_experts": getattr(tc, "num_experts", None),
        "hidden_size_per_layer_input": getattr(tc, "hidden_size_per_layer_input", None),
    }


def depth_fraction(layer: int, num_layers: int) -> float:
    """Normalised depth in [0, 1]: 0 at the first layer, 1 at the last.

    This is the only x-axis on which a 32-layer Qwen curve and a 60-layer
    Gemma curve can be drawn in the same panel.  `l / (L - 1)`, not `l / L`,
    so that both models' final layers — the ones the LI gate anchors — land
    on exactly 1.0.
    """
    if num_layers <= 1:
        return 0.0
    return layer / float(num_layers - 1)


# ---------------------------------------------------------------------------
# module-tree location
# ---------------------------------------------------------------------------
def _dig(obj, path: Sequence[str]):
    for attr in path:
        if not hasattr(obj, attr):
            return None
        obj = getattr(obj, attr)
    return obj


_LAYERS_PATHS = (
    ("model", "language_model", "layers"),   # Gemma4ForConditionalGeneration
    ("model", "layers"),                     # Gemma4ForCausalLM
    ("language_model", "model", "layers"),
    ("language_model", "layers"),
)
_NORM_PATHS = (
    ("model", "language_model", "norm"),
    ("model", "norm"),
    ("language_model", "model", "norm"),
    ("language_model", "norm"),
)


def locate(model) -> dict:
    """Find the decoder layer list, the final RMSNorm and the LM head.

    Raises rather than guessing: a wrong norm or a missing head produces a
    plausible-looking but meaningless curve, which is worse than a crash.
    """
    layers = norm = None
    layers_path = norm_path = None
    for p in _LAYERS_PATHS:
        got = _dig(model, p)
        if got is not None and hasattr(got, "__len__") and len(got) > 0:
            layers, layers_path = got, ".".join(p)
            break
    if layers is None:
        raise RuntimeError("could not locate the decoder layer list")

    for p in _NORM_PATHS:
        got = _dig(model, p)
        if got is not None and hasattr(got, "forward"):
            norm, norm_path = got, ".".join(p)
            break
    if norm is None:
        raise RuntimeError("could not locate the final RMSNorm")

    head = getattr(model, "lm_head", None)
    if head is None:
        raise RuntimeError("could not locate lm_head")

    return {"layers": layers, "norm": norm, "head": head,
            "layers_path": layers_path, "norm_path": norm_path,
            "n_modules": len(layers), "norm_class": type(norm).__name__}


def guard(model, config, *, expect_gemma: bool = True) -> dict:
    """Assert the checkpoint is the hybrid Gemma-4 we think it is, and that the
    module tree matches the config's depth.  Mirrors lib/model_guard.py's role
    on the Qwen side: no artifact is produced before this passes."""
    plan = layer_plan(config)
    loc = locate(model)
    problems: List[str] = []

    if loc["n_modules"] != plan["num_layers"]:
        problems.append(
            f"config says {plan['num_layers']} layers, module tree has "
            f"{loc['n_modules']}")
    arch = (getattr(config, "architectures", None) or [type(model).__name__])
    arch = list(arch)
    if expect_gemma and not any("gemma" in a.lower() for a in arch + [type(model).__name__]):
        problems.append(f"does not look like a Gemma checkpoint: {arch}")
    if plan["n_full_attn"] == 0:
        problems.append("no full-attention layers found — layer_types is wrong")
    if plan["layer_types"][-1] != FULL:
        # Gemma-4's config forces this; if it is violated the checkpoint is
        # unusual enough that the band summary would mislead.
        problems.append(f"last layer is {plan['layer_types'][-1]}, expected {FULL}")

    rec = {
        "architectures": arch,
        "model_class": type(model).__name__,
        "layers_path": loc["layers_path"],
        "norm_path": loc["norm_path"],
        "norm_class": loc["norm_class"],
        "problems": problems,
        **{k: v for k, v in plan.items() if k != "layer_types"},
        "layer_types_head": plan["layer_types"][:14],
    }
    if problems:
        raise RuntimeError(f"Gemma architecture guard failed: {problems}")
    return rec, plan, loc


def summarize(rec: dict) -> str:
    return (f"{rec['model_class']} | {rec['num_layers']} layers "
            f"({rec['n_sliding']} sliding + {rec['n_full_attn']} full, "
            f"{rec['layer_types_source']}) | H={rec['hidden_size']} | "
            f"softcap={rec['final_logit_softcapping']} | "
            f"norm={rec['norm_path']}:{rec['norm_class']}")


# ---------------------------------------------------------------------------
# the lens
# ---------------------------------------------------------------------------
_HEAD_CACHE: Dict[int, tuple] = {}


VOCAB_CHUNK = int(os.environ.get("MECH_LENS_VOCAB_CHUNK", "16384"))


def head_parts(head):
    """(weight, bias) of the LM head, left in their stored dtype and device.

    Deliberately NOT upcast here.  Gemma-4 ties lm_head to embed_tokens, so a
    full fp32 copy of this matrix is 262144 x 5376 x 4 B = 5.6 GiB -- which is
    fine on an idle 80 GB card and fatal when the model is sharded into
    whatever VRAM another job has left behind.  The lens upcasts per vocab
    chunk instead; see lens_token_logprobs.
    """
    w = head.weight.detach()
    b = head.bias.detach() if getattr(head, "bias", None) is not None else None
    return w, b


def head_fp32(head):
    """Full fp32 (W, b).  Only for tiny test heads -- see head_parts."""
    key = id(head)
    if key not in _HEAD_CACHE:
        w, b = head_parts(head)
        _HEAD_CACHE[key] = (w.float(), None if b is None else b.float())
    return _HEAD_CACHE[key]


def _normed(h_rows, norm):
    """Apply the model's final norm in fp32, on the norm's own device."""
    dev = getattr(getattr(norm, "weight", None), "device", None)
    h = h_rows if dev is None else h_rows.to(dev)
    # Gemma4RMSNorm internally casts to float and returns type_as(input), so
    # feeding it fp32 keeps the whole chain in fp32.
    return norm(h.float())


def lens_logits(h_rows, norm, head, softcap: Optional[float]):
    """The model's own readout, applied to an arbitrary layer's residual rows.

    h_rows: (n, H) raw (pre-final-norm) residual stream.
    Returns (n, V) fp32 logits, softcapped exactly as the model does.

    Materialises the whole fp32 vocab projection -- fine for tests, wasteful
    for a 262k vocab.  The runner uses lens_token_logprobs, which streams.
    """
    import torch
    W32, B32 = head_fp32(head)
    h = _normed(h_rows, norm).to(W32.device)
    logits = torch.nn.functional.linear(h, W32, B32)
    if softcap is not None:
        logits = torch.tanh(logits / softcap) * softcap
    return logits


def lens_token_logprobs(h_rows, targets, norm, head, softcap: Optional[float],
                        *, chunk: Optional[int] = None):
    """Per-token log-probs of `targets` read out of `h_rows`. -> list[float]

    Streams the vocab projection in chunks, upcasting each slice of the head to
    fp32 on the fly and accumulating the partition function with logaddexp.
    Peak extra memory is one (n, chunk) fp32 block plus one fp32 head slice,
    instead of a 5.6 GiB fp32 copy of the tied embedding matrix.

    fp32 is not optional: at a verb logit around 17 the bf16 ULP is 0.0625,
    which would quantise every log-prob we report.  Chunking is how we get it
    without the memory.
    """
    import torch
    W, B = head_parts(head)
    dev = W.device
    h = _normed(h_rows, norm).to(dev)                      # (n, H) fp32
    n, V = h.shape[0], W.shape[0]
    C = int(chunk or VOCAB_CHUNK)

    tgt = targets.to(dev).long()
    tgt_logit = torch.zeros(n, device=dev, dtype=torch.float32)
    lse = torch.full((n,), float("-inf"), device=dev, dtype=torch.float32)

    for s in range(0, V, C):
        e = min(s + C, V)
        Wc = W[s:e].float()
        Bc = None if B is None else B[s:e].float()
        lc = torch.nn.functional.linear(h, Wc, Bc)          # (n, e-s) fp32
        if softcap is not None:
            lc = torch.tanh(lc / softcap) * softcap
        lse = torch.logaddexp(lse, torch.logsumexp(lc, dim=-1))
        sel = (tgt >= s) & (tgt < e)
        if bool(sel.any()):
            idx = sel.nonzero(as_tuple=True)[0]
            tgt_logit[idx] = lc[idx, tgt[idx] - s]
        del Wc, Bc, lc

    return [float(v) for v in (tgt_logit - lse).tolist()]


# ---------------------------------------------------------------------------
# hidden-state capture
# ---------------------------------------------------------------------------
class LayerTap:
    """Capture the raw output of every decoder layer, sliced to `rows`.

    Why not `output_hidden_states=True`: see the module docstring — the last
    entry is silently replaced by the post-final-norm `last_hidden_state`, and
    re-norming it is wrong.  Hooking the layers keeps the semantics explicit
    and, because we slice inside the hook, drops peak memory from O(L*T*H) to
    O(L*n_rows*H).

    Usage:
        with LayerTap(loc["layers"], rows_idx) as tap:
            out = model(**inputs)
        h = tap.stack()            # {layer -> (n_rows, H) tensor}
    """

    def __init__(self, layers, rows_idx):
        self.layers = layers
        self.rows_idx = rows_idx
        self.captured: Dict[int, "object"] = {}
        self._handles: List[object] = []
        self._idx_by_dev: Dict[object, object] = {}

    def _idx_for(self, dev):
        """rows_idx on `dev`. A sharded model puts layers on several devices,
        and index_select refuses a cross-device index."""
        if dev not in self._idx_by_dev:
            self._idx_by_dev[dev] = (self.rows_idx if self.rows_idx.device == dev
                                     else self.rows_idx.to(dev))
        return self._idx_by_dev[dev]

    def _mk(self, li: int):
        def hook(module, args, output):
            t = output[0] if isinstance(output, tuple) else output
            if t is None or not hasattr(t, "index_select"):
                return
            if t.dim() == 3:
                t = t[0]
            # index_select copies, so the full [T, H] activation is free to go
            self.captured[li] = t.index_select(0, self._idx_for(t.device)).detach()
        return hook

    def __enter__(self):
        for li, layer in enumerate(self.layers):
            self._handles.append(layer.register_forward_hook(self._mk(li)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return False

    def stack(self) -> Dict[int, object]:
        missing = [i for i in range(len(self.layers)) if i not in self.captured]
        if missing:
            raise RuntimeError(f"no activation captured for layers {missing[:8]}"
                               f"{'...' if len(missing) > 8 else ''}")
        return self.captured


# ---------------------------------------------------------------------------
# CPU-only self test
# ---------------------------------------------------------------------------
def _selftest() -> int:
    class _TC:
        num_hidden_layers = 60
        hidden_size = 4096
        sliding_window = 512
        final_logit_softcapping = None
        tie_word_embeddings = True
        num_experts = None
        hidden_size_per_layer_input = 256
        layer_types = None

    class _C:
        def __init__(self, tc):
            self._tc = tc
            self.architectures = ["Gemma4ForConditionalGeneration"]

        def get_text_config(self):
            return self._tc

    fails = []

    # 1) derived 5:1 plan for a 60-layer model
    plan = layer_plan(_C(_TC()))
    if plan["num_layers"] != 60:
        fails.append(f"num_layers {plan['num_layers']}")
    if plan["full_attn_layers"] != [5, 11, 17, 23, 29, 35, 41, 47, 53, 59]:
        fails.append(f"full_attn_layers {plan['full_attn_layers']}")
    if (plan["n_full_attn"], plan["n_sliding"]) != (10, 50):
        fails.append(f"bands {plan['n_full_attn']}/{plan['n_sliding']} != 10/50")
    if plan["layer_types"][-1] != FULL:
        fails.append("last layer not full_attention")
    print(f"[selftest] derived plan: {plan['num_layers']} layers, "
          f"{plan['n_sliding']} sliding + {plan['n_full_attn']} full, "
          f"full at {plan['full_attn_layers']}")

    # 2) an explicit layer_types list must win over the derived pattern
    tc = _TC()
    tc.num_hidden_layers = 4
    tc.layer_types = [SLIDING, FULL, SLIDING, FULL]
    plan2 = layer_plan(_C(tc))
    if plan2["full_attn_layers"] != [1, 3] or plan2["layer_types_source"] != "config.layer_types":
        fails.append(f"explicit layer_types ignored: {plan2}")
    print(f"[selftest] explicit layer_types honoured: {plan2['layer_types_source']}")

    # 3) depth fraction: first layer 0.0, last layer exactly 1.0, both depths
    for L in (32, 60):
        if depth_fraction(0, L) != 0.0 or depth_fraction(L - 1, L) != 1.0:
            fails.append(f"depth_fraction endpoints wrong for L={L}")
    # Qwen L27 and Gemma L50 should land close together (0.871 vs 0.847)
    q, g = depth_fraction(27, 32), depth_fraction(50, 60)
    print(f"[selftest] depth_fraction: Qwen L27 -> {q:.3f}, Gemma L50 -> {g:.3f}")
    if not (0.0 <= q <= 1.0 and 0.0 <= g <= 1.0):
        fails.append("depth_fraction out of range")

    # 4) softcap replication is a pure function; check it against the formula
    try:
        import torch

        class _N:
            def __call__(self, x):
                return x

        class _H:
            weight = torch.eye(5)
            bias = None
        raw = torch.tensor([[3.0, -9.0, 0.5, 20.0, -1.0]])
        cap = 10.0
        got = lens_logits(raw, _N(), _H(), cap)
        want = torch.tanh(raw / cap) * cap
        if not torch.allclose(got, want, atol=1e-6):
            fails.append(f"softcap mismatch {got} vs {want}")
        # and that softcap=None is a no-op
        got0 = lens_logits(raw, _N(), _H(), None)
        if not torch.allclose(got0, raw, atol=1e-6):
            fails.append("softcap=None changed the logits")
        print(f"[selftest] softcap replication OK (cap={cap}); "
              f"None is a no-op")
        _HEAD_CACHE.clear()

        # 5) the chunked streaming log-softmax must equal the naive one, at
        #    every chunk size including ones that do not divide the vocab and
        #    ones smaller than the gap between targets
        torch.manual_seed(0)
        V, H, n = 997, 64, 5

        class _N2:
            weight = torch.ones(H)

            def __call__(self, x):
                return x

        class _H2:
            weight = torch.randn(V, H, dtype=torch.bfloat16)
            bias = None

        hh = torch.randn(n, H)
        tt = torch.tensor([0, 13, 500, 996, 500])
        for cap2 in (None, 30.0):
            ref = torch.log_softmax(lens_logits(hh, _N2(), _H2(), cap2), dim=-1)
            ref = ref.gather(1, tt[:, None]).squeeze(1)
            _HEAD_CACHE.clear()
            for C in (1, 7, 128, 997, 4096):
                got = torch.tensor(lens_token_logprobs(hh, tt, _N2(), _H2(),
                                                       cap2, chunk=C))
                if not torch.allclose(got, ref, atol=2e-5):
                    fails.append(f"chunked lens != naive at cap={cap2} chunk={C}: "
                                 f"max|d|={float((got-ref).abs().max()):.2e}")
                _HEAD_CACHE.clear()
        print(f"[selftest] chunked streaming log-softmax matches the naive "
              f"readout for chunk in 1..4096, softcap in (None, 30.0)")

        # 6) and it must not have upcast the whole head to do it
        W, _ = head_parts(_H2())
        if W.dtype != torch.bfloat16:
            fails.append(f"head_parts upcast the weight to {W.dtype}")
    except ImportError:
        print("[selftest] torch unavailable — skipped the softcap check")

    if fails:
        print("\n[selftest] FAIL:")
        for f in fails:
            print("   -", f)
        return 1
    print("\n[selftest] PASS")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--model", default=None,
                    help="print the layer plan for a real checkpoint's config.json")
    a = ap.parse_args()
    if a.model:
        import os
        from types import SimpleNamespace
        with open(os.path.join(a.model, "config.json")) as f:
            raw = json.load(f)
        tcd = raw.get("text_config", raw)
        tc = SimpleNamespace(**tcd)
        cfg = SimpleNamespace(architectures=raw.get("architectures"),
                              get_text_config=lambda: tc)
        print(json.dumps(layer_plan(cfg), indent=1)[:2000])
    raise SystemExit(_selftest() if a.selftest or not a.model else 0)
