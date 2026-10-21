"""
Lightweight paired statistics — sign test, Wilcoxon, bootstrap CI, Pearson,
Spearman. Kept minimal so we don't drag in scipy just for a few p-values.
"""
from __future__ import annotations
import math
from typing import Iterable, Sequence, Tuple

import numpy as np


def sign_test(diffs: Sequence[float]) -> Tuple[int, int, float]:
    """Two-sided exact sign test. Returns (n_pos, n_neg, p_two_sided)."""
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    n = pos + neg
    if n == 0:
        return pos, neg, 1.0
    from math import comb
    k = min(pos, neg)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return pos, neg, min(p, 1.0)


def bootstrap_mean_ci(xs: Sequence[float], n_boot: int = 2000, alpha: float = 0.05,
                     seed: int = 0) -> Tuple[float, float, float]:
    """Return (mean, low, high) percentile CI."""
    xs = np.asarray(xs, dtype=np.float64)
    if xs.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, xs.size, size=(n_boot, xs.size))
    means = xs[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(xs.mean()), float(lo), float(hi)


def pearson(x: Sequence[float], y: Sequence[float]) -> Tuple[float, int]:
    xs = np.asarray(x, dtype=np.float64); ys = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[mask], ys[mask]
    if xs.size < 2:
        return float("nan"), 0
    xs = xs - xs.mean(); ys = ys - ys.mean()
    denom = math.sqrt((xs * xs).sum() * (ys * ys).sum())
    if denom == 0:
        return float("nan"), int(xs.size)
    return float((xs * ys).sum() / denom), int(xs.size)


def spearman(x: Sequence[float], y: Sequence[float]) -> Tuple[float, int]:
    xs = np.asarray(x, dtype=np.float64); ys = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[mask], ys[mask]
    if xs.size < 2:
        return float("nan"), 0
    return pearson(_rank(xs), _rank(ys))


def _rank(xs: np.ndarray) -> np.ndarray:
    order = xs.argsort()
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(xs))
    # average ties
    _handle_ties(xs, ranks)
    return ranks


def _handle_ties(xs, ranks):
    order = xs.argsort()
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        if j > i:
            avg = (i + j) / 2.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
        i = j + 1
