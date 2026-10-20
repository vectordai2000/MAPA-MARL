"""Shared machinery for the experiment drivers.

One place that knows how to

* train a variant (resumable: an existing ``history.csv`` is reused),
* run a *greedy* evaluation roll-out and return per-step metrics, so that
  tables which need rates (collision rate, violation rate) do not have to
  re-implement the loop,
* aggregate across seeds with the paper's 95 % bootstrap protocol.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

from .baselines import build_config, make_agent
from .config import Config
from .env import UAVMECEnv
from .metrics import bootstrap_ci


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def train_variant(variant: str, profile: str, seed: int,
                  overrides: Optional[Dict[str, str]] = None,
                  episodes: Optional[int] = None, out: str = "results",
                  tag: Optional[str] = None, force: bool = False) -> Path:
    """Train (or reuse) one run and return its directory."""
    from .train import run

    tag = tag or variant
    run_dir = Path(out) / tag / f"seed{seed}"
    if (run_dir / "history.csv").exists() and not force:
        return run_dir
    cfg = build_config(variant, profile)
    if overrides:
        cfg.apply_overrides(overrides)
    if episodes is not None:
        cfg.train.episodes = episodes
    cfg.train.tag = tag
    run(cfg, seed, run_dir, verbose=False)
    return run_dir


def load_agent(variant: str, profile: str, seed: int,
               overrides: Optional[Dict[str, str]] = None,
               out: str = "results", tag: Optional[str] = None):
    """Rebuild the config/agent for a finished run and load its weights."""
    cfg = build_config(variant, profile)
    if overrides:
        cfg.apply_overrides(overrides)
    tag = tag or variant
    agent = make_agent(cfg.train.variant, cfg, cfg.train.device, seed)
    agent.load(str(Path(out) / tag / f"seed{seed}" / "model.pt"))
    return cfg, agent


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
def rollout_eval(agent, cfg: Config, seed: int, episodes: int = 3,
                 n_device: Optional[int] = None,
                 n_slot: Optional[int] = None,
                 burst_at: Optional[int] = None,
                 burst_devices: int = 0) -> Dict[str, np.ndarray]:
    """Greedy roll-out returning per-step metrics.

    ``burst_at`` / ``burst_devices`` implement the Fig. 9 scenario: extra devices
    become active after ``burst_at`` slots *within* the same episode.
    """
    ec = cfg.env
    if n_slot is not None:
        ec = _clone_env_cfg(cfg, n_slot=n_slot)

    rec: Dict[str, List[float]] = {}
    for ep in range(episodes):
        env = UAVMECEnv(ec, seed=seed + 10_000 + 97 * ep,
                        n_device=n_device or ec.n_device)
        agent.reset_hidden()
        obs = env.reset()
        while not env.done:
            if burst_at is not None and env.t == burst_at:
                env.activate_devices(burst_devices)
            action = agent.act(obs, explore=False)
            obs, info = env.step(action["speed"], action["direction"], action["mode"])
            agent.advance_hidden(obs["peer_act"])
            info = dict(info)
            info["t"] = env.t
            info["episode"] = ep
            info["n_device"] = env.n_device
            for k, v in info.items():
                rec.setdefault(k, []).append(float(v))
    return {k: np.asarray(v) for k, v in rec.items()}


def _clone_env_cfg(cfg: Config, **kwargs):
    import copy
    ec = copy.deepcopy(cfg.env)
    for k, v in kwargs.items():
        setattr(ec, k, v)
    return ec


def summarize_eval(rec: Dict[str, np.ndarray], rules: Iterable[str] = ()) -> Dict[str, float]:
    out = {
        "qos": float(rec["qos"].mean()),
        "jfi": float(rec["jfi"].mean()),
        "delay": float(rec["delay"].mean()),
        "energy_kj": float(rec["energy"].mean() * len(rec["energy"]) / max(rec["episode"].max() + 1, 1) / 1e3),
        "starvation": float(rec["starvation"].mean()),
        "collisions": float(rec["collisions"].mean()),
        "dist_violation_rate": float((rec["collisions"] > 0).mean()),
    }
    for rule in rules:                     # e.g. "violation=corridor" extensions
        out[rule] = float("nan")
    return out


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------
def aggregate_seeds(per_seed: List[Dict[str, float]], n_boot: int = 4000,
                    seed: int = 0) -> Dict[str, Dict[str, float]]:
    """Mean / std / 95 % bootstrapped CI over seeds for every scalar metric."""
    keys = [k for k in per_seed[0] if isinstance(per_seed[0][k], (int, float))]
    rng = np.random.default_rng(seed)
    out: Dict[str, Dict[str, float]] = {}
    for k in keys:
        vals = np.asarray([row[k] for row in per_seed], dtype=float)
        n = vals.size
        if n > 1:
            boots = vals[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
            lo, hi = np.quantile(boots, [0.025, 0.975])
        else:
            lo = hi = vals.mean()
        out[k] = {"mean": float(vals.mean()), "std": float(vals.std()),
                  "ci_low": float(lo), "ci_high": float(hi), "n": int(n)}
    return out


def save_json(path: str | Path, payload) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {p}")


__all__ = ["train_variant", "load_agent", "rollout_eval", "summarize_eval",
           "aggregate_seeds", "save_json"]
