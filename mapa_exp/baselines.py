"""Variant registry: the MAPA-MARL agent, its ablations and the baselines.

Every entry is a *flag combination* on the same code base, exactly as described
in Sec. V-B: baselines share the identical feature encoder and only differ in
how they handle the hybrid action space / the constraint.

| variant | agent class | what it is |
|---|---|---|
| ``mapa_marl``      | MAPA-MARL | ZGR + mask + PID-Lag. + opponent RNN |
| ``w_o_zgr``        | MAPA-MARL | discrete head trained by ST-GS (Table IV) |
| ``w_o_mask``       | MAPA-MARL | dense attention (Table IV) |
| ``w_o_pid``        | MAPA-MARL | plain dual ascent instead of PID-Lag. (Table IV) |
| ``mappo_native``   | MAPPO     | native categorical sampling (Fig. 5) |
| ``mappo_clipped``  | MAPPO     | native + independent optimiser + aggressive clipping |
| ``mappo_stgs``     | MAPPO     | straight-through Gumbel-Softmax, annealed temperature |
| ``maddpg``         | MAPA-MARL | off-policy DDPG backbone, ST-GS, no mask / PID / RNN |
| ``maac``           | MAPA-MARL | attention critic without the spatial prior (MAAC-style) |
| ``mrltl``          | MAPA-MARL | first-order meta-RL, no HVP curvature term |
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .config import Config


@dataclass(frozen=True)
class Variant:
    name: str
    agent: str                     # "mapa" | "mappo"
    overrides: Dict[str, Any]
    reference: str = ""            # which paper artefact it appears in


VARIANTS: Dict[str, Variant] = {
    "mapa_marl": Variant("mapa_marl", "mapa", {}),
    "w_o_zgr": Variant("w_o_zgr", "mapa", {"train.estimator": "stgs"}),
    "w_o_mask": Variant("w_o_mask", "mapa", {"train.use_mask": False}),
    "w_o_pid": Variant("w_o_pid", "mapa", {"train.use_pid": False}),
    "mappo_native": Variant("mappo_native", "mappo", {"train.estimator": "native"}),
    "mappo_clipped": Variant("mappo_clipped", "mappo",
                             {"train.estimator": "native_clipped"}),
    "mappo_stgs": Variant("mappo_stgs", "mappo", {"train.estimator": "stgs"}),
    "maddpg": Variant("maddpg", "mapa",
                      {"train.estimator": "stgs", "train.use_mask": False,
                       "train.use_pid": False, "train.use_rnn": False}),
    "maac": Variant("maac", "mapa",
                    {"train.estimator": "stgs", "train.use_pid": False,
                     "train.use_rnn": False}),
    "mrltl": Variant("mrltl", "mapa",
                     {"train.use_meta": True, "train.meta_hvp_coef": 0.0}),
}

DISPLAY = {
    "mapa_marl": "MAPA-MARL",
    "w_o_zgr": "w/o ZGR",
    "w_o_mask": "w/o Sparse-Mask",
    "w_o_pid": "w/o PID-Lagrangian",
    "mappo_native": "MAPPO (native)",
    "mappo_clipped": "MAPPO (clipped)",
    "mappo_stgs": "MAPPO (ST-GS)",
    "maddpg": "MADDPG",
    "maac": "MAAC",
    "mrltl": "MRLTO",
}


def build_config(variant: str, profile: str = "paper") -> Config:
    if variant not in VARIANTS:
        raise KeyError(f"unknown variant {variant!r}; known: {sorted(VARIANTS)}")
    cfg = Config.load(profile)
    cfg.apply_overrides(VARIANTS[variant].overrides)
    cfg.train.variant = variant
    cfg.train.tag = variant
    return cfg


def make_agent(variant: str, cfg: Config, device: Optional[str] = None, seed: int = 0):
    """Instantiate the agent class that implements ``variant``."""
    if variant not in VARIANTS:
        raise KeyError(f"unknown variant {variant!r}")
    if VARIANTS[variant].agent == "mappo":
        from .mappo import MAPPOAgent
        return MAPPOAgent(cfg, device=device, seed=seed)
    from .agent import MAPAAgent
    return MAPAAgent(cfg, device=device, seed=seed)


__all__ = ["VARIANTS", "DISPLAY", "Variant", "build_config", "make_agent"]
