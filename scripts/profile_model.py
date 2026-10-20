"""Model profiler: parameter count, arithmetic cost and inference latency.

::

    python scripts/profile_model.py --profile paper
    python scripts/profile_model.py --profile quick --device cpu
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mapa_exp.agent import MAPAAgent, obs_to_tensors  # noqa: E402
from mapa_exp.config import Config  # noqa: E402
from mapa_exp.env import UAVMECEnv  # noqa: E402


def count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def flops_per_forward(agent: MAPAAgent, cfg: Config, device: torch.device) -> float:
    """Arithmetic cost of one onboard forward pass (actor + RNN), in MFLOPs."""
    env = UAVMECEnv(cfg.env, seed=0)
    obs = env.reset()
    o = obs_to_tensors(obs, cfg, device)
    ctx = torch.zeros(1, cfg.env.n_uav, cfg.train.rnn_hidden, device=device)
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except Exception:                                             # pragma: no cover
        return float("nan")
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        agent.actor(o, ctx)
    return counter.get_total_flops() / 1e6


def latency(agent: MAPAAgent, cfg: Config, device: torch.device,
            iters: int = 50) -> dict:
    env = UAVMECEnv(cfg.env, seed=0)
    obs = env.reset()
    o = obs_to_tensors(obs, cfg, device)
    ctx = torch.zeros(1, cfg.env.n_uav, cfg.train.rnn_hidden, device=device)
    for _ in range(5):
        agent.actor(o, ctx)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            agent.actor(o, ctx)
    if device.type == "cuda":
        torch.cuda.synchronize()
    forward = (time.perf_counter() - t0) / iters * 1e3

    for _ in range(5):
        agent.act(obs, explore=False)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        agent.act(obs, explore=False)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return {"forward_ms": forward,
            "decision_ms": (time.perf_counter() - t0) / iters * 1e3}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="paper", choices=["paper", "quick"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/profile.json")
    args = ap.parse_args()

    cfg = Config.load(args.profile)
    device = torch.device(args.device if (torch.cuda.is_available() or
                                          args.device == "cpu") else "cpu")
    agent = MAPAAgent(cfg, device=str(device), seed=0)

    actor_params = count_params(agent.actor) + count_params(agent.opp)
    critic_params = count_params(agent.critic)
    flops = flops_per_forward(agent, cfg, device)
    lat = latency(agent, cfg, device)

    print(f"profile={args.profile}  device={device}  "
          f"N={cfg.env.n_uav}  M={cfg.env.n_device}\n")
    print(f"  actor + RNN params      {actor_params / 1e6:8.3f} M  "
          f"({actor_params * 4 / 1024 ** 2:.1f} MB, FP32)")
    print(f"  critic params           {critic_params / 1e6:8.3f} M  "
          f"({critic_params * 4 / 1024 ** 2:.1f} MB, FP32)")
    print(f"  actor forward           {flops:8.1f} MFLOPs")
    print(f"  actor forward latency   {lat['forward_ms']:8.2f} ms/step")
    print(f"  full decision path      {lat['decision_ms']:8.2f} ms/step")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "profile": args.profile, "device": str(device),
        "actor_params": actor_params, "critic_params": critic_params,
        "actor_mflops": flops, **lat}, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
