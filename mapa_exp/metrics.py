"""Statistics used by the experiment drivers.

* 95 % **bootstrapped** confidence intervals over seeds (the paper's protocol,
  Sec. V-A), obtained by resampling whole seed runs with replacement;
* the "episodes to convergence" definition used in Table IV;
* a paired comparison helper so that "ours vs baseline" is not judged from
  overlapping error bars.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np


def bootstrap_ci(matrix: np.ndarray, n_boot: int = 10_000, alpha: float = 0.05,
                 seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean curve with a bootstrapped CI band.

    Parameters
    ----------
    matrix : [n_seeds, n_episodes] per-seed curves (equal length).

    Returns
    -------
    (mean, lower, upper) arrays of shape [n_episodes].
    """
    matrix = np.asarray(matrix, dtype=float)
    n_seeds, n_ep = matrix.shape
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_seeds, size=(n_boot, n_seeds))
    boots = matrix[idx].mean(axis=1)                     # [n_boot, n_ep]
    return (matrix.mean(axis=0),
            np.quantile(boots, alpha / 2, axis=0),
            np.quantile(boots, 1 - alpha / 2, axis=0))


def smooth(x: np.ndarray, window: int = 21) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if window <= 1 or x.size < window:
        return x
    pad = window // 2
    padded = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")[: x.size]


def episodes_to_threshold(curve: np.ndarray, tol: float = 0.05,
                          window: int = 21) -> int:
    """First episode after which the smoothed curve stays inside a ±``tol`` band.

    The band is relative to the curve's own final value, which is the definition
    used for the "Episodes" column of Table IV: how long the run needs before it
    no longer improves by more than 5 % of its final value.
    """
    c = smooth(curve, window)
    final = float(np.mean(c[-max(5, c.size // 20):]))
    band = abs(final) * tol
    outside = np.abs(c - final) > band
    if not outside.any():
        return 1
    return int(np.where(outside)[0][-1]) + 2


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 10_000,
                     seed: int = 0) -> Dict[str, float]:
    """Paired difference of two per-seed metric vectors (same seeds)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    diff = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n_boot, diff.size))
    boots = diff[idx].mean(axis=1)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {"mean_diff": float(diff.mean()), "ci_low": float(lo),
            "ci_high": float(hi),
            "significant": bool(lo > 0 or hi < 0)}


def relative_deviation(measured: float, reported: float) -> float:
    if reported == 0:
        return float("nan")
    return (measured - reported) / abs(reported) * 100.0


__all__ = ["bootstrap_ci", "smooth", "episodes_to_threshold", "paired_bootstrap",
           "relative_deviation"]
