"""Train the agent for a few minutes and plot the learning curve.

    python examples/train_demo.py --episodes 80

Writes ``results/demo/{history,eval}.csv`` and ``results/demo/curve.png``.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mapa_exp.config import Config
from mapa_exp.train import run


def plot(csv_path: Path, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with csv_path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    ep = np.array([float(r["episode"]) for r in rows])
    rew = np.array([float(r["reward"]) for r in rows])
    pad = 5
    smooth = np.convolve(np.pad(rew, (pad, pad), mode="edge"),
                         np.ones(2 * pad + 1) / (2 * pad + 1), mode="valid")

    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    ax.plot(ep, rew, color="#9fb6cc", lw=0.9, label="episodic reward")
    ax.plot(ep, smooth, color="#c0392b", lw=1.8, label="moving average")
    ax.set_xlabel("episode")
    ax.set_ylabel("episodic reward")
    ax.set_title("Training on the UAV-MEC environment", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, lw=0.6)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/demo")
    args = ap.parse_args()

    cfg = Config.load("quick")
    cfg.train.episodes = args.episodes
    out_dir = Path(args.out)
    result = run(cfg, args.seed, out_dir, verbose=True)
    print(f"\nfinished in {result['seconds']:.0f}s")
    plot(out_dir / "history.csv", out_dir / "curve.png")


if __name__ == "__main__":
    main()
