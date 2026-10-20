"""Device benchmark: latency, throughput, precision and memory.

::

    python scripts/benchmark.py                       # cuda if available
    python scripts/benchmark.py --device cpu
    python scripts/benchmark.py --profile paper --batch-sizes 1 8 32

Measures, with warm-up and proper synchronisation:

* per-step inference latency (p50 / p95) for batch sizes 1 .. 32,
* throughput in decisions/second,
* FP32 vs FP16 (autocast) forward cost,
* optional ``torch.compile`` speed-up (``--compile``),
* GPU memory allocated / reserved / peak,
* environment roll-out throughput (simulator, numpy).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mapa_exp.agent import MAPAAgent, obs_to_tensors  # noqa: E402
from mapa_exp.config import Config  # noqa: E402
from mapa_exp.cuda import (amp_context, device_info, maybe_compile,  # noqa: E402
                           memory_report, reset_peak_memory, resolve_device,
                           synchronize, timed)
from mapa_exp.env import UAVMECEnv  # noqa: E402


def make_obs(cfg: Config, device: torch.device, batch: int) -> dict:
    obs = UAVMECEnv(cfg.env, seed=0).reset()
    tensors = obs_to_tensors(obs, cfg, device)
    return {k: v.expand(batch, *v.shape[1:]).contiguous() for k, v in tensors.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="paper", choices=["paper", "quick"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32])
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--out", default="results/benchmark.json")
    args = ap.parse_args()

    cfg = Config.load(args.profile)
    device = resolve_device(args.device)
    print(f"device: {json.dumps(device_info(device))}\n")

    agent = MAPAAgent(cfg, device=str(device), seed=0)
    agent.actor = maybe_compile(agent.actor, args.compile).eval()
    ctx0 = torch.zeros(1, cfg.env.n_uav, cfg.train.rnn_hidden, device=device)

    results = {"profile": args.profile, "device": str(device),
               "compiled": args.compile, "batch_sizes": {}}

    print(f"{'batch':>6}{'fp32 p50':>10}{'fp32 p95':>10}{'fp16 p50':>10}"
          f"{'decisions/s':>14}")
    print("-" * 50)
    for batch in args.batch_sizes:
        obs = make_obs(cfg, device, batch)
        ctx = ctx0.expand(batch, -1, -1).contiguous()
        reset_peak_memory(device)

        def forward32():
            with torch.no_grad():
                agent.actor(obs, ctx)

        def forward16():
            with torch.no_grad(), amp_context(device, True):
                agent.actor(obs, ctx)

        fp32 = timed(forward32, device, args.iters)
        fp16 = timed(forward16, device, args.iters)
        per_decision = fp32["mean_ms"] / batch
        print(f"{batch:>6}{fp32['p50_ms']:>10.2f}{fp32['p95_ms']:>10.2f}"
              f"{fp16['p50_ms']:>10.2f}{1000 / per_decision:>14.0f}")
        results["batch_sizes"][str(batch)] = {
            "fp32": fp32, "fp16": fp16,
            "ms_per_decision": per_decision,
            "decisions_per_second": 1000 / per_decision,
        }

    # ---- full decision path: numpy observation -> action -------------------
    obs_np = UAVMECEnv(cfg.env, seed=0).reset()

    def decide():
        agent.act(obs_np, explore=False)

    dec = timed(decide, device, args.iters)
    print(f"\ndecision path (obs -> action)  {dec['p50_ms']:7.2f} ms p50   "
          f"{1000 / dec['p50_ms']:7.0f} decisions/s")
    results["decision_path"] = dec

    # ---- environment roll-out throughput (host-side simulator) --------------
    env = UAVMECEnv(cfg.env, seed=1)
    env.reset()
    steps = 0
    synchronize(device)
    import time
    t0 = time.perf_counter()
    while steps < 200:
        speed = torch.rand(cfg.env.n_uav).numpy()
        direction = torch.randn(cfg.env.n_uav, 2).numpy()
        mode = torch.randint(0, 3, (cfg.env.n_uav,)).numpy()
        _, _ = env.step(speed, direction, mode)
        steps += 1
        if env.done:
            env.reset()
    env_sps = steps / (time.perf_counter() - t0)

    mem = memory_report(device)
    print(f"\nenvironment roll-out      {env_sps:8.0f} steps/s "
          f"({cfg.env.n_uav} UAVs, {cfg.env.n_device} devices)")
    if mem:
        print(f"GPU memory allocated      {mem['allocated_mb']:8.1f} MB")
        print(f"GPU memory reserved       {mem['reserved_mb']:8.1f} MB")
        print(f"GPU memory peak           {mem['peak_mb']:8.1f} MB")
    else:
        print("GPU memory                n/a (CPU device)")

    results["env_steps_per_second"] = env_sps
    results["memory"] = mem
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
