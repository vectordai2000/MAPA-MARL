"""Configuration for the paper-reproduction experiments.

Two layers of parameters:

* **reported** -- values stated in the manuscript (Sec. V-A, Table V): N, M,
  area, learning rates, PID gains, seeds, episode budget, RNN width, TBPTT.
* **calibrated** -- constants the manuscript does not state (task size,
  ``T_max``, ``P_hov``, ``a/b/alpha``, network width).  They are chosen by
  ``scripts/calibrate_operating_point.py`` so that the *reported operating
  point* (Table III: latency 1.24 s, energy 19.8 kJ, QoS 91.8 % at N = 5,
  M = 50) is reached.  Every calibrated value is listed in
  ``docs/assumptions.md`` and marked ``# [calibrated]`` below.

Profiles
--------
``paper``  N=5, M=50 (padded to 80), S=40, 1200 episodes, 10 seeds.
``quick``  a 10-20x cheaper setting used to verify the pipeline end-to-end.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Sequence


@dataclass
class EnvConfig:
    # --- geometry & time ------------------------------------------------
    area: float = 500.0                 # reported
    height: float = 100.0               # reported
    n_uav: int = 5                      # reported (default)
    n_device: int = 50                  # reported (default)
    n_device_max: int = 80              # sweep upper bound (Fig. 7)
    n_slot: int = 40                    # [calibrated] from 19.8 kJ / episode
    dt: float = 0.5                     # reported (2 Hz macro planning)
    v_max: float = 15.0                 # reported (Sec. V-H)
    d_min: float = 10.0                 # reported (Table II discussion)
    battery: float = 1.0e5

    # --- air-to-ground channel (Eq. 1-2) --------------------------------
    bandwidth: float = 1.0e6
    tx_power: float = 0.1
    noise_dbm: float = -114.0
    beta0_db: float = -60.0             # [calibrated]
    alpha: float = 3.0                  # [calibrated]
    a_los: float = 9.61                 # [calibrated] standard urban constant
    b_los: float = 0.16                 # [calibrated] standard urban constant
    mu_nlos_db: float = 20.0            # [calibrated]

    # --- computation & energy (Eq. 3-4) ---------------------------------
    task_min: float = 3.25e5            # [calibrated]
    task_max: float = 9.75e5            # [calibrated]
    cycles_per_bit: float = 800.0       # [calibrated]
    cpu_local: float = 0.35e9           # [calibrated] IoT-class CPU: LOCAL must not dominate
    cpu_uav: float = 8.0e9              # [calibrated]
    cpu_cloud: float = 20.0e9           # [calibrated]
    p_uav_compute: float = 25.0         # [calibrated]
    p_tx: float = 3.0                   # [calibrated]
    backhaul_rate: float = 15.0e6       # [calibrated] keeps EDGE vs CLOUD non-degenerate
    backhaul_delay: float = 0.03        # [calibrated]
    p_hover: float = 170.0              # [calibrated] -> 19.8 kJ / episode (Table III)
    kappa_drag: float = 0.01            # [calibrated]

    # --- QoS / fairness / objective (Eq. 5-8) ---------------------------
    t_max: float = 1.50                 # [calibrated]
    theta: float = 3.0                  # [calibrated]
    w_delay: float = 0.5                # reported
    w_energy: float = 0.5               # reported
    energy_ref: float = 100.0           # J per UAV per slot (Table III: 19.8 kJ / 40 slots / 5 UAVs)
    jfi_threshold: float = 0.85         # paper uses J_th; exact value not stated
    safety_weight: float = 0.3
    device_speed: float = 1.5

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrainConfig:
    # --- experiment scale ------------------------------------------------
    episodes: int = 1200                # reported
    seeds: Sequence[int] = field(default_factory=lambda: tuple(range(10)))  # reported
    eval_every: int = 20
    eval_episodes: int = 3
    profile: str = "paper"

    # --- network width (Table V targets: 4.5 M actor, 15.2 M critic) -----
    d_model: int = 288                  # [calibrated] -> ~4.5 M actor params (Table V)
    n_heads: int = 8
    n_layers: int = 6
    critic_d_model: int = 448           # [calibrated] -> ~15.2 M critic params
    critic_n_layers: int = 9
    rnn_hidden: int = 128               # reported
    tbptt: int = 16                     # reported

    # --- RL (reported: lr 3e-4 / 1e-3) -----------------------------------
    gamma: float = 0.95
    tau: float = 0.01
    lr_actor: float = 3e-4              # reported
    lr_actor_disc: float = 3e-4
    lr_critic: float = 1e-3             # reported
    lr_opponent: float = 1e-3
    batch_size: int = 256
    buffer_episodes: int = 200
    warmup_steps: int = 2000
    update_every: int = 1
    grad_clip: float = 5.0
    entropy_coef: float = 0.01
    explore_dir_std: float = 0.6
    explore_speed_std: float = 0.1
    explore_temp: float = 1.0

    # --- discrete gradient estimator -------------------------------------
    estimator: str = "zgr"              # zgr | native | stgs
    zgr_temperature: float = 0.1
    stgs_tau_start: float = 5.0
    stgs_tau_end: float = 0.1

    # --- PID-Lagrangian (reported gains) ---------------------------------
    use_pid: bool = True
    k_p: float = 0.1                    # reported
    k_i: float = 0.01                   # reported
    k_d: float = 0.05                   # reported
    lambda_init: float = 0.5
    lambda_max: float = 10.0
    ema_beta: float = 0.9               # reported
    ema_delta: float = 0.02
    ema_kappa: float = 25.0

    # --- spatial mask -----------------------------------------------------
    use_mask: bool = True
    mask_sigma_d: float = 150.0
    mask_gamma: float = 2.0
    mask_tau: float = 8.0
    mask_eps_th: float = 0.35

    # --- opponent model / meta layer -------------------------------------
    use_rnn: bool = True
    use_meta: bool = False
    meta_init_path: str = ""
    meta_iters: int = 200
    meta_tasks: int = 4
    meta_inner_steps: int = 8
    meta_lr: float = 1e-4               # reported
    meta_eta: float = 2e-3
    meta_hvp_coef: float = 0.1
    meta_delay: int = 1
    meta_gac_eta: float = 0.5

    # --- scenario (Fig. 9: sudden task burst) -----------------------------
    burst_episode: int = -1             # episode at which extra devices appear
    burst_devices: int = 0

    # --- bookkeeping ------------------------------------------------------
    device: str = "cuda"
    out_dir: str = "runs"
    tag: str = "full"
    variant: str = "mapa_marl"
    log_every: int = 20
    verbose: bool = True

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["seeds"] = list(self.seeds)
        return d


QUICK_OVERRIDES: Dict[str, Any] = {
    "env.n_uav": 3,
    "env.n_device": 12,
    "env.n_device_max": 16,
    "env.n_slot": 20,
    "train.episodes": 100,
    "train.eval_every": 10,
    "train.seeds": (0, 1),
    "train.d_model": 64,
    "train.n_heads": 4,
    "train.n_layers": 2,
    "train.critic_d_model": 64,
    "train.critic_n_layers": 2,
    "train.rnn_hidden": 32,
    "train.batch_size": 128,
    "train.warmup_steps": 400,
    "train.buffer_episodes": 60,
}


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Config":
        env = EnvConfig(**{k: v for k, v in payload.get("env", {}).items()
                           if k in {f.name for f in fields(EnvConfig)}})
        tr = TrainConfig(**{k: v for k, v in payload.get("train", {}).items()
                            if k in {f.name for f in fields(TrainConfig)}})
        tr.seeds = tuple(tr.seeds)
        return cls(env=env, train=tr)

    @classmethod
    def from_json(cls, path: str | Path) -> "Config":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def load(cls, profile: str = "paper") -> "Config":
        """Build the config for a named profile (``paper`` or ``quick``)."""
        cfg = cls()
        if profile == "quick":
            cfg.apply_overrides(QUICK_OVERRIDES)
            cfg.train.profile = "quick"
        elif profile == "paper":
            cfg.train.profile = "paper"
        else:
            raise ValueError(f"unknown profile: {profile}")
        return cfg

    def to_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps({"env": self.env.to_dict(), "train": self.train.to_dict()},
                       indent=2), encoding="utf-8")

    def apply_overrides(self, overrides: Dict[str, Any]) -> "Config":
        for key, value in overrides.items():
            section, _, name = key.partition(".")
            if not name:
                raise KeyError(f"override needs '<env|train>.<field>': {key}")
            target = getattr(self, section, None)
            if target is None or not hasattr(target, name):
                raise KeyError(f"unknown override: {key}")
            setattr(target, name, _coerce(value, getattr(target, name)))
        return self


def _coerce(value: Any, current: Any) -> Any:
    if isinstance(current, bool):
        return value if isinstance(value, bool) else str(value).lower() in {"1", "true", "yes", "on"}
    if isinstance(current, int) and not isinstance(current, bool):
        return int(value)
    if isinstance(current, float):
        return float(value)
    if isinstance(current, tuple):
        if isinstance(value, (list, tuple)):
            return tuple(int(v) if isinstance(current[0], int) else float(v) for v in value)
        return tuple(int(v) if isinstance(current[0], int) else float(v)
                     for v in str(value).strip("()").split(","))
    return value
