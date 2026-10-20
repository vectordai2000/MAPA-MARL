"""PID-Lagrangian constraint handling for the fairness CMDP.

Paper, Sec. III-E, Eq. (9)-(11).  The constraint is ``E[J] >= J_th`` with Jain's
fairness index ``J``.  The constraint error is ``e_t = J_th - J(s_t)`` (>0 means
violation) and the multiplier follows

.. code-block:: text

    beta_t = beta_base * exp(-kappa * relu(|e_t - e_{t-1}| - delta_th))   (9)
    e~_t   = beta_t e~_{t-1} + (1 - beta_t) e_t                           (10)
    lambda_{t+1} = relu(lambda_t + Kp e_t + Ki sum_tau e_tau + Kd (e~_t - e~_{t-1}))  (11)

The adaptive EMA short-circuits (``beta_t -> 0``) when the fairness error moves
faster than ``delta_th``: the derivative term then acts on the *instantaneous*
error, which is the "feed-forward compensation" that removes the lag of a
plain EMA during a sudden violation.

``PIDLagrangian`` is also the ablation harness: with ``use_pid=False`` the same
object degenerates into the standard first-order dual ascent
``lambda <- relu(lambda + eta e_t)`` that the paper shows oscillating.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import math


@dataclass
class PIDLagrangian:
    """Adaptive EMA-filtered PID multiplier (Eq. 9-11)."""

    jfi_threshold: float = 0.85
    k_p: float = 0.1
    k_i: float = 0.01
    k_d: float = 0.05
    lambda_init: float = 0.5
    lambda_max: float = 5.0
    ema_beta: float = 0.9          # beta_base
    ema_delta: float = 0.02        # delta_th
    ema_kappa: float = 25.0        # kappa
    use_pid: bool = True
    integral_max: float = 40.0     # anti-windup clamp

    lam: float = field(init=False)
    integral: float = field(init=False, default=0.0)
    ema: float = field(init=False, default=0.0)
    ema_prev: float = field(init=False, default=0.0)
    err_prev: float = field(init=False, default=0.0)
    beta: float = field(init=False, default=0.9)
    history: Dict[str, List[float]] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.lam = float(min(max(self.lambda_init, 0.0), self.lambda_max))
        self.history = {"lambda": [], "ema": [], "beta": [], "error": []}

    # ------------------------------------------------------------------
    @property
    def value(self) -> float:
        return self.lam

    def update(self, jfi: float) -> float:
        """Consume the fairness index of the current step and return lambda."""
        e = float(self.jfi_threshold) - float(jfi)

        # adaptive EMA (Eq. 9-10)
        jump = abs(e - self.err_prev)
        beta = self.ema_beta * math.exp(-self.ema_kappa * max(0.0, jump - self.ema_delta))
        self.beta = beta
        self.ema = beta * self.ema + (1.0 - beta) * e

        d_term = self.ema - self.ema_prev
        self.ema_prev = self.ema
        self.err_prev = e

        if self.use_pid:
            # anti-windup: freeze the integrator at the actuator limits
            at_limit = (self.lam >= self.lambda_max and e > 0) or (self.lam <= 0.0 and e < 0)
            if not at_limit:
                self.integral = float(
                    min(max(self.integral + e, -self.integral_max), self.integral_max)
                )
            delta = self.k_p * e + self.k_i * self.integral + self.k_d * d_term
        else:
            # ablation: plain first-order dual ascent
            delta = self.k_p * e

        self.lam = float(min(max(self.lam + delta, 0.0), self.lambda_max))
        self._log(e)
        return self.lam

    def _log(self, e: float) -> None:
        self.history["lambda"].append(self.lam)
        self.history["ema"].append(self.ema)
        self.history["beta"].append(self.beta)
        self.history["error"].append(e)

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.lam = float(min(max(self.lambda_init, 0.0), self.lambda_max))
        self.integral = 0.0
        self.ema = 0.0
        self.ema_prev = 0.0
        self.err_prev = 0.0
        self.history = {"lambda": [], "ema": [], "beta": [], "error": []}

    def state_dict(self) -> Dict[str, float]:
        return {"lambda": self.lam, "integral": self.integral,
                "ema": self.ema, "ema_prev": self.ema_prev,
                "err_prev": self.err_prev}

    def load_state_dict(self, state: Dict[str, float]) -> None:
        for k, v in state.items():
            setattr(self, k, v)


__all__ = ["PIDLagrangian"]
