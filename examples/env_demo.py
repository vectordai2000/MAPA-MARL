"""Environment walkthrough: roll out a hand-written policy and print the metrics.

    python examples/env_demo.py

No learning happens here — the point is to show the simulator API: step with
``(speed, direction, mode)``, read ``info`` (latency, QoS, fairness, energy).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mapa_exp import UAVMECEnv
from mapa_exp.config import Config

POLICIES = ("local", "edge", "cloud", "edge-goto")


def main() -> None:
    cfg = Config.load("quick")
    print(f"N={cfg.env.n_uav}  M={cfg.env.n_device}  S={cfg.env.n_slot}  "
          f"dt={cfg.env.dt}s  area={cfg.env.area}m\n")
    print(f"{'policy':<12}{'delay[s]':>9}{'QoS':>7}{'JFI':>7}{'energy[J/slot]':>16}")
    print("-" * 51)
    for policy in POLICIES:
        rows = []
        for ep in range(3):
            env = UAVMECEnv(cfg.env, seed=100 + ep)
            env.reset()
            while not env.done:
                speed, direction, mode = env.heuristic_action(policy)
                _, info = env.step(speed, direction, mode)
                rows.append(info)
        mean = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        print(f"{policy:<12}{mean['delay']:>9.3f}{mean['qos']:>7.3f}"
              f"{mean['jfi']:>7.3f}{mean['energy']:>16.1f}")

    env = UAVMECEnv(cfg.env, seed=7)
    obs = env.reset()
    print(f"\nobservation keys: {sorted(obs)}")
    print(f"  self        {obs['self'].shape}")
    print(f"  peer        {obs['peer'].shape}")
    print(f"  dev         {obs['dev'].shape}")
    print(f"  dist_uu     {obs['dist_uu'].shape}   (spatial mask input)")


if __name__ == "__main__":
    main()
