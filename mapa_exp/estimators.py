"""Discrete-action gradient estimators for the hybrid action space.

The discrete head has to produce a gradient for a categorical decision that is
embedded in a critic input.  Three options are relevant to the paper:

1. native categorical sampling + REINFORCE (huge variance, see Fig. 6),
2. temperature-annealed straight-through Gumbel-Softmax (ST-GS) -- Eq. (13) is
   *not* used, the gradient flows through the hard sample, which explodes as
   tau -> 0,
3. the **Variance-Bounded Zero-temperature Gumbel-Rao (ZGR)** proxy of the
   paper: a Rao-Blackwellised, state-dependent baseline is subtracted from the
   candidate values, which keeps the gradient in the same scale as the
   continuous DPG stream.

The ZGR surrogate implemented here is (Eq. 13)::

    L_disc = - sum_i P(k_i | z_n) * ( Q(s, a_n=k_i) - B(z_n) )        (13)
    B(z_n) = sum_i P(k_i | z_n) * Q(s, a_n=k_i)                         (baseline)

with ``Q`` detached (score-function estimator) and ``P`` the softmax at a small
"zero-temperature" relaxation ``tau_zgr``.  Its gradient w.r.t. the logits is
``sum_i P_i (Q_i - B) dlogP_i``, i.e. exactly the policy gradient of the
categorical head with a variance-absorbing baseline.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

N_MODES = 3


def categorical_entropy(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    probs = torch.softmax(logits / temperature, dim=-1)
    return -(probs * torch.log(probs + 1e-8)).sum(-1)


def zgr_surrogate(logits: torch.Tensor, q_candidates: torch.Tensor,
                  temperature: float = 0.1) -> torch.Tensor:
    """Eq. (13): Rao-Blackwellised ZGR loss (minimise to ascend Q).

    Parameters
    ----------
    logits        : [*, 3] categorical logits of the discrete head.
    q_candidates  : [*, 3] critic values Q(s, a_n = k_i), detached internally.
    temperature   : small relaxation temperature (the ``zero-temperature``
                    limit is emulated with tau=0.1 to keep ``P`` well defined).

    Returns
    -------
    scalar loss; its gradient w.r.t. ``logits`` is the variance-bounded
    empirical gradient proxy used to update the discrete head.
    """
    probs = torch.softmax(logits / temperature, dim=-1)
    q = q_candidates.detach()
    baseline = (probs.detach() * q).sum(-1, keepdim=True)      # B(z_n)
    advantage = q - baseline
    return -(probs * advantage).sum(-1).mean()


def stgs_one_hot(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Straight-through Gumbel-Softmax sample (baseline / ablation)."""
    return F.gumbel_softmax(logits, tau=max(temperature, 1e-3), hard=True, dim=-1)


def anneal_temperature(step: int, total: int, start: float = 5.0,
                       end: float = 0.1) -> float:
    """Exponential temperature schedule used by the ST-GS baseline (Sec. V-B)."""
    frac = min(max(step / max(total, 1), 0.0), 1.0)
    return max(end, start * (end / start) ** frac)


def zgr_bias_bound(q_candidates: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """Empirical proxy of the local curvature ||H_Q||_inf (diagnostic only).

    The paper bounds the ZGR bias by ``0.5 * ||H_Q||_inf * Var(z_n)``.  We report
    the spread of the candidate values as a cheap, model-free curvature proxy.
    """
    probs = torch.softmax(logits.detach(), dim=-1)
    mean = (probs * q_candidates.detach()).sum(-1, keepdim=True)
    var = (probs * (q_candidates.detach() - mean) ** 2).sum(-1)
    return var.sqrt().mean()


__all__ = ["zgr_surrogate", "stgs_one_hot", "categorical_entropy",
           "anneal_temperature", "zgr_bias_bound", "N_MODES"]
