"""Spatial regularisers: the Sparsity-Induced Inverse-Distance Soft-Mask.

Paper, Sec. IV-B, Eq. (14)-(15)::

    rho_{n,j} = 1 / (1 + (d_{n,j} / sigma_d) ** gamma)              (14)
    M_{n,j}   = (1/tau) * log(1 + exp(tau * (rho_{n,j} - eps_th)))    (15)   [= softplus]

The mask is not a gating trick: it is used as an additive bias on the attention
logits, ``e~ = q^T k / sqrt(d) + log(M + delta)``, so that physically decoupled
agents contribute neither value nor gradient to the local attention output.
``softplus`` is C^inf (twice differentiable), unlike the hard ReLU truncation
used by standard sparse attention, which removes the gradient-fault problem.

For comparison, the two baselines used by the paper are provided as well:

* ``rbf_mask``   -- the standard Gaussian RBF soft-mask (dense graph, heavy-tail
  truncation that underflows instead of producing true sparsity).
* ``dense``      -- no spatial prior at all (fully connected attention).
"""

from __future__ import annotations

import numpy as np
import torch


# ---------------------------------------------------------------------------
# numpy / environment side
# ---------------------------------------------------------------------------
def inverse_distance_rho(dist, sigma_d: float = 150.0, gamma: float = 2.0):
    """Raw inverse-distance spatial correlation, Eq. (14)."""
    dist = np.asarray(dist, dtype=np.float64)
    return 1.0 / (1.0 + (dist / float(sigma_d)) ** float(gamma))


def softplus_mask(dist, sigma_d: float = 150.0, gamma: float = 2.0,
                  tau: float = 8.0, eps_th: float = 0.35):
    """Smooth sparsity-induced mask, Eq. (15), returned in (0, 1].

    Implemented with ``np.logaddexp`` for numerical stability at large ``tau``.
    """
    rho = inverse_distance_rho(dist, sigma_d, gamma)
    return np.logaddexp(0.0, tau * (rho - eps_th)) / tau


def rbf_mask(dist, sigma_d: float = 150.0, gamma: float = 2.0,
             tau: float = 8.0, eps_th: float = 0.35):
    """Gaussian-RBF baseline ``exp(-d^2 / (2 sigma_d^2))`` (dense, no truncation)."""
    dist = np.asarray(dist, dtype=np.float64)
    return np.exp(-0.5 * (dist / float(sigma_d)) ** 2)


# ---------------------------------------------------------------------------
# torch side (used by the networks)
# ---------------------------------------------------------------------------
def log_mask_bias(mask: torch.Tensor, delta: float = 1e-6,
                  floor: float = -20.0) -> torch.Tensor:
    """Turn a soft mask in (0, 1] into an additive attention-logit bias."""
    return torch.log(mask.clamp_min(delta)).clamp_min(floor)


def mask_from_distance(dist: torch.Tensor, sigma_d: float = 150.0,
                       gamma: float = 2.0, tau: float = 8.0,
                       eps_th: float = 0.35) -> torch.Tensor:
    """Differentiable version of Eq. (14)-(15) on a distance tensor."""
    rho = 1.0 / (1.0 + (dist / sigma_d).clamp_min(0.0) ** gamma)
    return torch.nn.functional.softplus(tau * (rho - eps_th)) / tau


__all__ = [
    "inverse_distance_rho",
    "softplus_mask",
    "rbf_mask",
    "log_mask_bias",
    "mask_from_distance",
]
