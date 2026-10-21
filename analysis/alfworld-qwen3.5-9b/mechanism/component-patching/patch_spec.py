"""
component-patching shared specification — the ONLY definition of what gets patched.

Model-agnostic by construction. Both runners (Qwen `run_component_patch.py`,
Gemma `run_component_patch.py`) import this file, and the Gemma tree's
copy is a symlink to this one, so the two models cannot drift apart by editing a
constant. Anything architecture-specific lives in the runner that owns it.

See README.md for the rationale; every number here is documented there.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

# ── exposure contrast ────────────────────────────────────────────────────────
K_HI = 8          # donor
K_LO = 0          # host
INFO_CONDITION = "neutral"

# ── depth bands: fixed a priori, identical for both models ───────────────────
BAND_EDGES = (0.0, 0.25, 0.50, 0.75, 1.0)
N_BANDS = len(BAND_EDGES) - 1

# ── statistics ───────────────────────────────────────────────────────────────
N_BOOT = 4000
SEED = 0

# ── gates ────────────────────────────────────────────────────────────────────
SELF_TOL = 1e-6          # self-patch must be exactly zero to this tolerance
CEILING_TOL = 1e-3       # resid_last recovery must be 1.0 to this tolerance
REFERENCE_TOL = 0.05     # nats: single-forward vs single-forward, bf16 level


# ══════════════════════════════════════════════════════════════════════════
# Site naming
# ══════════════════════════════════════════════════════════════════════════

def mix_site(l: int) -> str:
    """Token-mixing module output. Qwen: linear_attn (GDN) or self_attn.
    Gemma: self_attn (sliding or full). Same name, same definition."""
    return f"mix_L{l:02d}"


def mlp_site(l: int) -> str:
    return f"mlp_L{l:02d}"


def resid_site(l: int) -> str:
    """Residual stream leaving decoder layer l (whole-layer output)."""
    return f"resid_L{l:02d}"


def postmix_site(l: int) -> str:
    """GEMMA ONLY: post_attention_layernorm output = the tensor actually added
    to the residual at the mixer site. On Qwen this equals mix_site(l) exactly,
    which is why the headline uses the module output instead (README §2.1)."""
    return f"postattn_L{l:02d}"


def postmlp_site(l: int) -> str:
    """GEMMA ONLY: post_feedforward_layernorm output."""
    return f"postffn_L{l:02d}"


EMBED_SITE = "embed"


def depth_frac(l: int, num_layers: int) -> float:
    """d = l / (L-1). Qwen: l/31. Gemma: l/59. The only depth coordinate the
    main text is allowed to use."""
    if num_layers < 2:
        raise ValueError("num_layers must be >= 2")
    return l / (num_layers - 1)


def depth_bands(num_layers: int) -> List[List[int]]:
    """Partition layers into N_BANDS fixed relative-depth bands.

    Band b holds every layer with BAND_EDGES[b] <= d < BAND_EDGES[b+1], except
    the last band which is closed at d = 1.0 so the final layer is never
    orphaned. Edges are fixed before any result is seen and are the same for
    both models; the band *sizes* differ (8/8/8/8 on L=32, 15/15/15/15 on L=60)
    because depth is normalised, not because layers were chosen.
    """
    bands: List[List[int]] = [[] for _ in range(N_BANDS)]
    for l in range(num_layers):
        d = depth_frac(l, num_layers)
        b = N_BANDS - 1
        for i in range(N_BANDS):
            lo, hi = BAND_EDGES[i], BAND_EDGES[i + 1]
            if (lo <= d < hi) or (i == N_BANDS - 1 and d >= lo):
                b = i
                break
        bands[b].append(l)
    if sum(len(b) for b in bands) != num_layers:
        raise AssertionError("band partition lost layers")
    if any(not b for b in bands):
        raise AssertionError(f"empty band for L={num_layers}: {bands}")
    return bands


# ══════════════════════════════════════════════════════════════════════════
# Arms
# ══════════════════════════════════════════════════════════════════════════

def _arm(name, sites, donor, kind, component=None, scope="both", **extra) -> dict:
    return {"name": name, "sites": sorted(set(sites)), "donor": donor,
            "kind": kind, "component": component, "scope": scope, **extra}


def build_arms(num_layers: int, layer_types: Sequence[str], arch: str,
               stage: str = "all") -> List[dict]:
    """The complete arm list, in run order.

    arch: "qwen" | "gemma" — selects the arch-specific Appendix arms and
    validates that layer_types uses that architecture's vocabulary.
    stage: "headline" (Figure b), "bands" (Appendix depth localisation),
           "all" (both plus arch-specific).

    Donors: "hi" = the same state's k=8 activations, "self" = the host's own
    k=0 activations (control, must be exactly 0), "rand" = a different game's
    k=8 activations (control, must not recover).
    """
    if arch not in ("qwen", "gemma"):
        raise ValueError(f"unknown arch {arch!r}")
    L = num_layers
    mix_all = [mix_site(l) for l in range(L)]
    mlp_all = [mlp_site(l) for l in range(L)]

    arms = [
        # scale anchors and controls first: if any of these fails the cell is
        # worthless and we want to know before spending time on the sweep.
        _arm("embed", [EMBED_SITE], "hi", "control",
             expect="exactly 0: action tokens are identical by construction"),
        _arm("self_all", mix_all + mlp_all, "self", "control",
             expect="|delta| < 1e-6: the harness must reproduce its own forward"),
        _arm("resid_last", [resid_site(L - 1)], "hi", "anchor",
             expect="recovery == 1.000: the final residual determines the logits"),
        # headline
        _arm("all_mix", mix_all, "hi", "headline", component="token_mixer"),
        _arm("all_mlp", mlp_all, "hi", "headline", component="mlp"),
        # random-state controls
        _arm("rand_all_mix", mix_all, "rand", "control", component="token_mixer",
             expect="must not stably recover the exposure effect"),
        _arm("rand_all_mlp", mlp_all, "rand", "control", component="mlp",
             expect="must not stably recover the exposure effect"),
    ]

    if stage in ("bands", "all"):
        bands = depth_bands(L)
        for b, layers in enumerate(bands):
            arms.append(_arm(f"mix_band{b}", [mix_site(l) for l in layers], "hi",
                             "band", component="token_mixer", band=b,
                             layers=list(layers),
                             depth_range=[BAND_EDGES[b], BAND_EDGES[b + 1]]))
            arms.append(_arm(f"mlp_band{b}", [mlp_site(l) for l in layers], "hi",
                             "band", component="mlp", band=b, layers=list(layers),
                             depth_range=[BAND_EDGES[b], BAND_EDGES[b + 1]]))
        # Leave-one-band-out. `*_band{b}` alone tests SUFFICIENCY (can this depth
        # range carry the effect by itself?); `*_all_but_band{b}` tests
        # NECESSITY (is this range required?). The pair is what makes the depth
        # localisation informative, because `all_mix` on its own is pinned at the
        # structural ceiling (README §5.1) and cannot localise anything.
        for b, layers in enumerate(bands):
            skip = set(layers)
            arms.append(_arm(f"mix_all_but_band{b}",
                             [mix_site(l) for l in range(L) if l not in skip], "hi",
                             "band_loo", component="token_mixer", band=b,
                             layers=[l for l in range(L) if l not in skip],
                             depth_range=[BAND_EDGES[b], BAND_EDGES[b + 1]]))
            arms.append(_arm(f"mlp_all_but_band{b}",
                             [mlp_site(l) for l in range(L) if l not in skip], "hi",
                             "band_loo", component="mlp", band=b,
                             layers=[l for l in range(L) if l not in skip],
                             depth_range=[BAND_EDGES[b], BAND_EDGES[b + 1]]))

    if arch == "qwen" and stage == "all":
        lt = list(layer_types)
        if not set(lt) <= {"linear_attention", "full_attention"}:
            raise ValueError(f"qwen expects linear/full_attention layer types, got {set(lt)}")
        full = [l for l, t in enumerate(lt) if t == "full_attention"]
        lin = [l for l, t in enumerate(lt) if t == "linear_attention"]
        arms.append(_arm("all_mix_full", [mix_site(l) for l in full], "hi",
                         "arch_specific", component="token_mixer", scope="qwen",
                         layers=full, note="8 softmax-attention layers"))
        arms.append(_arm("all_mix_lin", [mix_site(l) for l in lin], "hi",
                         "arch_specific", component="token_mixer", scope="qwen",
                         layers=lin, note="24 Gated DeltaNet layers"))

    if arch == "gemma" and stage == "all":
        lt = list(layer_types)
        if not set(lt) <= {"sliding_attention", "full_attention"}:
            raise ValueError(f"gemma expects sliding/full_attention layer types, got {set(lt)}")
        arms.append(_arm("all_mix_postnorm", [postmix_site(l) for l in range(L)], "hi",
                         "arch_specific", component="token_mixer", scope="gemma",
                         note="post_attention_layernorm output: bounds the sandwich-norm "
                              "magnitude attenuation of README §2.1"))
        arms.append(_arm("all_mlp_postnorm", [postmlp_site(l) for l in range(L)], "hi",
                         "arch_specific", component="mlp", scope="gemma",
                         note="post_feedforward_layernorm output"))

    seen = set()
    for a in arms:
        if a["name"] in seen:
            raise AssertionError(f"duplicate arm {a['name']}")
        seen.add(a["name"])
    return arms


def required_sites(arms: Sequence[dict], num_layers: int, arch: str) -> List[str]:
    """Every site name any arm refers to — the runner must install a hook for
    each of these, and nothing else needs recording."""
    sites = set()
    for a in arms:
        sites.update(a["sites"])
    return sorted(sites)


# ══════════════════════════════════════════════════════════════════════════
# Random-donor map (shared by both trees, generated once)
# ══════════════════════════════════════════════════════════════════════════

def build_donor_order(entries: Sequence[dict], seed: int = SEED) -> List[str]:
    """Deterministic processing order such that each state's random donor (the
    previous state in this order) comes from a DIFFERENT game.

    `entries` = [{"state_id":..., "game_name":...}]. Greedy: walk a seeded
    permutation and only accept a state as the next element if its game differs
    from the current tail's game; deferred states are retried at the end. If no
    valid order exists (one game holding >50% of states) the leftover states are
    appended and flagged, and the runner records `rand_donor_game_mismatch=True`
    for them rather than silently using a same-game donor.
    """
    import random

    game_of = {e["state_id"]: e["game_name"] for e in entries}
    ids = sorted(game_of)
    rng = random.Random(seed)
    rng.shuffle(ids)

    order: List[str] = []
    pending = list(ids)
    while pending:
        placed = False
        for i, sid in enumerate(pending):
            if not order or game_of[sid] != game_of[order[-1]]:
                order.append(sid)
                pending.pop(i)
                placed = True
                break
        if not placed:
            # every remaining state shares the tail's game: append the rest and
            # let the runner flag them.
            order.extend(pending)
            break
    return order


def donor_partner(order: Sequence[str], index: int) -> Tuple[Optional[str], bool]:
    """-> (donor_state_id or None, game_mismatch_flag)."""
    if index == 0:
        return None, False
    return order[index - 1], False


def donor_map_path() -> str:
    """ONE file for both models, living in the Qwen tree.

    The Gemma tree's `lib/config.py` already hard-points ANALYSIS_ROOT at the
    Qwen tree for exactly this reason: model-independent artifacts must not be
    duplicated, or they drift. The random-state control is only a control if both
    models are handed the SAME mismatched donors, so the map is generated once
    here and read from here by both runners. Override with EXP_M_DONOR_MAP only
    to relocate it, never to regenerate it per model.
    """
    env = os.environ.get("EXP_M_DONOR_MAP")
    if env:
        return env
    # Both trees carry an identical copy of this file, so resolve to the Qwen
    # tree explicitly (the original used one absolute path for both models);
    # a path relative to this copy would give the Gemma run its own map.
    return os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
        "alfworld-qwen3.5-9b", "mechanism", "results", "component-patching",
        "donor_map.json"))


def ensure_donor_map(entries: Sequence[dict], arch: str) -> Tuple[List[str], dict]:
    """Load the shared map, or create it on first use and verify it thereafter.

    If the file already exists its order must reproduce exactly from the same
    seed and state set; a mismatch means the two models would be given different
    random donors, which silently destroys the control, so it is a hard error
    rather than a regeneration.
    """
    path = donor_map_path()
    order = build_donor_order(entries, SEED)
    game_of = {e["state_id"]: e["game_name"] for e in entries}
    if os.path.exists(path):
        existing = load_donor_map(path)
        if existing.get("order") == list(order):
            return order, existing
        raise RuntimeError(
            f"{path} exists with a different order ({len(existing.get('order', []))} "
            f"states vs {len(order)}). Both models must share one donor map; "
            f"delete it deliberately if the state set really changed.")
    return order, write_donor_map(path, order, game_of, f"component-patching/{arch}")


def write_donor_map(path: str, order: Sequence[str], game_of: Dict[str, str],
                    arch_note: str = "") -> dict:
    payload = {
        "seed": SEED,
        "n_states": len(order),
        "n_games": len(set(game_of.values())),
        "generated_by": arch_note or "component-patching",
        "rule": "donor = previous state in this order; game-disjoint by construction; "
                "the first state of each shard has no random arm",
        "order": list(order),
        "game_of": {s: game_of[s] for s in order},
        "donor_of": {order[i]: order[i - 1] for i in range(1, len(order))},
    }
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)
    os.replace(tmp, path)
    return payload


def load_donor_map(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════════
# Metric: ratio of means with a game-clustered bootstrap
# ══════════════════════════════════════════════════════════════════════════

def _game_means(rows: Sequence[Tuple[str, float]]) -> Tuple[List[str], List[float]]:
    by_game: Dict[str, List[float]] = defaultdict(list)
    for g, v in rows:
        by_game[g].append(float(v))
    games = sorted(by_game)
    return games, [float(sum(by_game[g]) / len(by_game[g])) for g in games]


def clustered_mean_ci(rows: Sequence[Tuple[str, float]], n_boot: int = N_BOOT,
                      seed: int = SEED) -> dict:
    """Game-clustered CI for a MEAN. rows = [(game, value)]."""
    import numpy as np

    if not rows:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n": 0, "n_games": 0}
    games, gm = _game_means(rows)
    arr = np.asarray(gm, dtype=np.float64)
    point = float(arr.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boots = arr[idx].mean(axis=1)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {"point": point, "lo": float(lo), "hi": float(hi),
            "n": len(rows), "n_games": len(games)}


def clustered_ratio_ci(num: Sequence[Tuple[str, float]],
                       den: Sequence[Tuple[str, float]],
                       n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Game-clustered CI for a RATIO OF MEANS: mean(R) / mean(G).

    Never a mean of per-state ratios — states with a near-zero denominator
    diverge there. The ratio is recomputed inside every bootstrap sample, so the
    CI respects that the denominator is itself estimated. `num` and `den` must
    cover the same states; they are intersected on (game, index) by requiring
    equal length and identical game order, which the caller guarantees by
    building both from the same per-state loop.
    """
    import numpy as np

    if len(num) != len(den):
        raise ValueError(f"numerator/denominator mismatch: {len(num)} vs {len(den)}")
    if not num:
        return {"ratio_pct": float("nan"), "lo_pct": float("nan"),
                "hi_pct": float("nan"), "n": 0, "n_games": 0,
                "mean_num": float("nan"), "mean_den": float("nan")}
    if [g for g, _ in num] != [g for g, _ in den]:
        raise ValueError("num and den must be game-aligned state by state")

    games, num_gm = _game_means(num)
    _, den_gm = _game_means(den)
    n_arr = np.asarray(num_gm, dtype=np.float64)
    d_arr = np.asarray(den_gm, dtype=np.float64)

    def ratio(nm, dm):
        n = float(np.mean(nm))
        d = float(np.mean(dm))
        return (100.0 * n / d) if abs(d) > 1e-12 else float("nan")

    point = ratio(n_arr, d_arr)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_arr.size, size=(n_boot, n_arr.size))
    boots = np.array([ratio(n_arr[i], d_arr[i]) for i in idx])
    boots = boots[np.isfinite(boots)]
    lo, hi = (np.quantile(boots, [0.025, 0.975]) if boots.size
              else (float("nan"), float("nan")))
    return {"ratio_pct": point, "lo_pct": float(lo), "hi_pct": float(hi),
            "n": len(num), "n_games": len(games),
            "mean_num": float(n_arr.mean()), "mean_den": float(d_arr.mean()),
            "n_boot_finite": int(boots.size), "n_boot": int(n_boot)}


def sign_test(diffs: Sequence[float]) -> dict:
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    n = pos + neg
    if n == 0:
        return {"pos": 0, "neg": 0, "p": float("nan")}
    from math import comb
    k = min(pos, neg)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return {"pos": pos, "neg": neg, "p": min(p, 1.0)}


# ══════════════════════════════════════════════════════════════════════════
# Record schema — what a per-cell JSON must contain for the analyzer to run
# ══════════════════════════════════════════════════════════════════════════

REQUIRED_CELL_FIELDS = (
    "model", "arch", "state_id", "game_name", "modality", "hi_k", "lo_k",
    "num_layers", "action_start_hi", "action_start_lo", "delta_T",
    "n_action_tokens", "checks", "logp_hi", "logp_lo", "gap", "arms",
)

REQUIRED_CHECKS = (
    "stem_prefix_ok", "action_ids_match", "embed_null_ok", "self_patch_ok",
    "ceiling_ok",
)

# arms whose failure invalidates the cell rather than just the arm
GATE_ARMS = ("embed", "self_all", "resid_last")


def cell_gate_ok(rec: dict) -> Tuple[bool, List[str]]:
    """-> (ok, reasons). A cell that fails is dropped by the analyzer and the
    drop is reported; gates are never silently relaxed."""
    reasons = []
    for f in REQUIRED_CELL_FIELDS:
        if f not in rec:
            reasons.append(f"missing field {f}")
    if reasons:
        return False, reasons
    ck = rec.get("checks", {})
    for c in REQUIRED_CHECKS:
        if c not in ck:
            reasons.append(f"missing check {c}")
        elif ck[c] is not True:
            reasons.append(f"check {c} = {ck[c]}")
    for name in GATE_ARMS:
        if name not in rec.get("arms", {}):
            reasons.append(f"missing gate arm {name}")
    return (not reasons), reasons
