"""Training / evaluation entry point.

Single-seed run::

    python -m mapa_exp.train --profile quick --episodes 100 --seed 0
    python -m mapa_exp.train --profile paper --set train.variant=w_o_zgr

See ``examples/`` for complete, self-contained scripts.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from .agent import MAPAAgent
from .baselines import make_agent
from .config import Config
from .env import UAVMECEnv

METRIC_KEYS = ("cost", "jfi", "qos", "starvation", "delay", "energy",
               "collisions", "safety", "mode_local", "mode_edge", "mode_cloud")


# ---------------------------------------------------------------------------
def evaluate(agent: MAPAAgent, cfg: Config, seed: int, episodes: int,
             task_scale: float = 1.0, n_device: Optional[int] = None) -> Dict[str, float]:
    """Greedy evaluation rollouts (no exploration noise, no learning).

    ``n_device`` defaults to the configured device count but is overridden by the
    task-burst experiment so that evaluation reflects the current swarm scenario.
    """
    acc: Dict[str, List[float]] = {k: [] for k in METRIC_KEYS}
    acc["reward"] = []
    for ep in range(episodes):
        env = UAVMECEnv(cfg.env, seed=seed + 10_000 + 97 * ep,
                        n_device=n_device or cfg.env.n_device,
                        task_scale=task_scale)
        agent.reset_hidden()
        obs = env.reset()
        ret = 0.0
        while not env.done:
            action = agent.act(obs, explore=False)
            obs, info = env.step(action["speed"], action["direction"], action["mode"])
            agent.advance_hidden(obs["peer_act"])
            lam = agent.lagrangian.value
            ret += (-(info["cost"] + cfg.env.safety_weight * info["safety"])
                    + lam * (info["jfi"] - cfg.env.jfi_threshold))
            for k in METRIC_KEYS:
                acc[k].append(info[k])
        acc["reward"].append(ret)
    return {k: float(np.mean(v)) for k, v in acc.items()}


def run(cfg: Config, seed: int, out_dir: Path, verbose: bool = True) -> Dict:
    """Train one agent for one seed and return its history."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = cfg.train.device
    env = UAVMECEnv(cfg.env, seed=seed, n_device=cfg.env.n_device)
    agent = make_agent(cfg.train.variant, cfg, device=device, seed=seed)

    if cfg.train.meta_init_path:
        init = Path(cfg.train.meta_init_path)
        if init.exists():
            agent.load(str(init))
            if verbose:
                print(f"[seed {seed}] loaded meta-initialisation from {init}")

    out_dir.mkdir(parents=True, exist_ok=True)
    hist_path = out_dir / "history.csv"
    eval_path = out_dir / "eval.csv"
    trace_path = out_dir / "trace.csv"
    hist_keys = ["episode", "reward", "lambda"] + list(METRIC_KEYS) + \
                ["critic_loss", "actor_loss", "grad_disc", "grad_cont", "opp_loss"]
    steps_per_episode = cfg.env.n_slot
    total_steps = cfg.train.episodes * steps_per_episode
    if total_steps < cfg.train.warmup_steps and cfg.train.variant not in (
            "mappo_native", "mappo_clipped", "mappo_stgs"):
        print(f"WARNING: {total_steps} environment steps < warmup_steps="
              f"{cfg.train.warmup_steps}: this run performs **no gradient update**. "
              f"Increase --episodes or lower --set train.warmup_steps.")

    rows: List[Dict[str, float]] = []
    eval_rows: List[Dict[str, float]] = []
    trace: List[Dict[str, float]] = []
    t0 = time.time()
    step_count = 0

    for episode in range(1, cfg.train.episodes + 1):
        if episode == cfg.train.burst_episode and cfg.train.burst_devices > 0:
            added = env.activate_devices(cfg.train.burst_devices)
            if verbose:
                print(f"[seed {seed}] task burst: +{added} devices at episode {episode}")

        obs = env.reset()
        agent.begin_episode()
        step_info: List[Dict[str, float]] = []
        updates: List[Dict[str, float]] = []
        ret = 0.0

        while not env.done:
            lam = agent.lagrangian.value
            action = agent.act(obs, explore=True)
            next_obs, info = env.step(action["speed"], action["direction"],
                                      action["mode"])
            ret += (-(info["cost"] + cfg.env.safety_weight * info["safety"])
                    + lam * (info["jfi"] - cfg.env.jfi_threshold))
            agent.push(obs, action, info, next_obs, env.done)
            agent.lagrangian.update(info["jfi"])       # Eq. (11)
            trace.append({"episode": episode, "step": step_count,
                          "lambda": agent.lagrangian.value,
                          "jfi": info["jfi"], "cost": info["cost"],
                          "mode": action["mode"].mean()})
            obs = next_obs
            step_info.append(info)
            step_count += 1
            if step_count % cfg.train.update_every == 0:
                m = agent.update()
                if m is not None:
                    updates.append(m)

        ppo_metrics = agent.end_episode()
        if isinstance(ppo_metrics, dict):
            # on-policy baselines update at the end of an episode, not per step
            updates.append({"critic_loss": ppo_metrics.get("v_loss", float("nan")),
                            "actor_loss": ppo_metrics.get("pi_loss", float("nan")),
                            "grad_disc": float("nan"), "grad_cont": float("nan"),
                            "opp_loss": float("nan")})
        row = {"episode": episode, "reward": ret,
               "lambda": agent.lagrangian.value}
        for k in METRIC_KEYS:
            row[k] = float(np.mean([s[k] for s in step_info]))
        for k in ("critic_loss", "actor_loss", "grad_disc", "grad_cont", "opp_loss"):
            vals = [u[k] for u in updates if u[k] == u[k]]
            row[k] = float(np.mean(vals)) if vals else float("nan")
        rows.append(row)

        if episode % cfg.train.eval_every == 0 or episode == 1:
            ev = evaluate(agent, cfg, seed, cfg.train.eval_episodes,
                          n_device=env.n_device)
            ev["episode"] = episode
            eval_rows.append(ev)
            if verbose:
                print(f"[seed {seed}] ep {episode:4d} | R {ret:8.2f} | "
                      f"QoS {row['qos']:.3f} | JFI {row['jfi']:.3f} | "
                      f"lam {row['lambda']:.2f} | evalQoS {ev['qos']:.3f} | "
                      f"t {time.time() - t0:5.1f}s")

    _write_csv(hist_path, rows, hist_keys)
    _write_csv(eval_path, eval_rows, ["episode", "reward"] + list(METRIC_KEYS))
    _write_csv(trace_path, trace, ["episode", "step", "lambda", "jfi", "cost", "mode"])
    agent.save(str(out_dir / "model.pt"))
    cfg.to_json(out_dir / "config.json")

    return {"history": rows, "eval": eval_rows, "seconds": time.time() - t0}


def _write_csv(path: Path, rows: List[Dict[str, float]], keys: List[str]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, float("nan")) for k in keys})


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Train MAPA-MARL on the paper environment.")
    ap.add_argument("--config", type=str, default=None, help="JSON config file")
    ap.add_argument("--profile", type=str, default="quick",
                    choices=["quick", "paper"],
                    help="quick = pipeline verification, paper = the reported setting")
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    ap.add_argument("--tag", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                    help="dotted override, e.g. --set train.use_mask=false")
    return ap


def config_from_args(args: argparse.Namespace) -> Config:
    if args.config:
        cfg = Config.from_json(args.config)
    else:
        cfg = Config.load(args.profile)
    overrides = {}
    for item in args.set:
        key, _, value = item.partition("=")
        overrides[key.strip()] = value.strip()
    cfg.apply_overrides(overrides)
    if args.episodes is not None:
        cfg.train.episodes = args.episodes
    if args.device is not None:
        cfg.train.device = args.device
    if args.tag is not None:
        cfg.train.tag = args.tag
    if args.out is not None:
        cfg.train.out_dir = args.out
    return cfg


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = config_from_args(args)
    out_dir = Path(cfg.train.out_dir) / cfg.train.tag / f"seed{args.seed}"
    result = run(cfg, args.seed, out_dir, verbose=not args.quiet)
    print(json.dumps({"tag": cfg.train.tag, "seed": args.seed,
                      "seconds": round(result["seconds"], 1),
                      "last": {k: round(v, 4) for k, v in
                               result["history"][-1].items()}}, indent=2))


if __name__ == "__main__":
    main()
