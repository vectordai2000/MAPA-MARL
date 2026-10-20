"""UAV-MEC multi-agent RL toolkit: hybrid action spaces, constrained MARL.

Public API
----------
``Config`` / ``EnvConfig`` / ``TrainConfig``   configuration
``UAVMECEnv``                                  the simulator
``MAPAAgent``                                  the hybrid-action agent
``MAPPOAgent``                                 on-policy PPO baseline
``make_agent`` / ``build_config`` / ``VARIANTS``  variant factory
``obs_to_tensors``                             observation encoding
"""

from .agent import MAPAAgent, obs_to_tensors
from .baselines import DISPLAY, VARIANTS, build_config, make_agent
from .config import Config, EnvConfig, TrainConfig
from .env import CLOUD, EDGE, LOCAL, MODE_NAMES, UAVMECEnv
from .mappo import MAPPOAgent

__version__ = "0.2.0"

__all__ = [
    "Config", "EnvConfig", "TrainConfig",
    "UAVMECEnv", "LOCAL", "EDGE", "CLOUD", "MODE_NAMES",
    "MAPAAgent", "MAPPOAgent", "obs_to_tensors",
    "make_agent", "build_config", "VARIANTS", "DISPLAY",
    "__version__",
]
