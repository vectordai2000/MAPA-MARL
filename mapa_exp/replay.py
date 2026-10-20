"""Episode replay buffer.

Transitions are stored per episode because the implicit opponent model needs
contiguous action histories (TBPTT windows), while the actor/critic updates
need a flat uniform minibatch.  Both views are kept in sync by rebuilding a
flattened cache whenever an episode is closed.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional

import numpy as np

_SCALARS = ("cost", "jfi", "safety", "done", "act_mode", "peer_act",
            "opp_ctx", "opp_ctx_next")
_VECTORS = ("act_cont",)


class EpisodeBuffer:
    def __init__(self, max_episodes: int = 60, seed: int = 0,
                 warmup_steps: int = 400):
        self.eps: deque = deque(maxlen=max_episodes)
        self.rng = np.random.default_rng(seed)
        self.warmup_steps = warmup_steps
        self._cur: Optional[Dict[str, List]] = None
        self._flat: Optional[Dict] = None
        self._n_flat = 0

    # ------------------------------------------------------------------
    def begin_episode(self) -> None:
        self._cur = {"obs": [], "next_obs": [], **{k: [] for k in _SCALARS},
                     **{k: [] for k in _VECTORS}}

    def push(self, obs: Dict[str, np.ndarray], act_cont: np.ndarray, act_mode: np.ndarray,
             info: Dict[str, float], next_obs: Dict[str, np.ndarray], done: bool,
             opp_ctx: np.ndarray, opp_ctx_next: np.ndarray) -> None:
        assert self._cur is not None, "call begin_episode() first"
        self._cur["obs"].append(obs)
        self._cur["next_obs"].append(next_obs)
        self._cur["act_cont"].append(np.asarray(act_cont, dtype=np.float32))
        self._cur["act_mode"].append(np.asarray(act_mode, dtype=np.int64))
        self._cur["cost"].append(info["cost"])
        self._cur["jfi"].append(info["jfi"])
        self._cur["safety"].append(info["safety"])
        self._cur["done"].append(float(done))
        self._cur["opp_ctx"].append(np.asarray(opp_ctx, dtype=np.float32))
        self._cur["opp_ctx_next"].append(np.asarray(opp_ctx_next, dtype=np.float32))
        self._cur["peer_act"].append(np.asarray(next_obs["peer_act"], dtype=np.float32))

    def end_episode(self) -> None:
        if self._cur is None or not self._cur["cost"]:
            self._cur = None
            return
        ep: Dict[str, object] = {}
        for split in ("obs", "next_obs"):
            keys = self._cur[split][0].keys()
            ep[split] = {k: np.stack([o[k] for o in self._cur[split]]) for k in keys}
        for k in _SCALARS:
            ep[k] = np.asarray(self._cur[k])
        for k in _VECTORS:
            ep[k] = np.stack(self._cur[k])
        self.eps.append(ep)
        self._cur = None
        self._flat = None

    # ------------------------------------------------------------------
    def _rebuild(self) -> None:
        if self._flat is not None:
            return
        if not self.eps:
            self._flat, self._n_flat = {}, 0
            return
        flat: Dict[str, object] = {}
        for k in _SCALARS:
            flat[k] = np.concatenate([ep[k] for ep in self.eps])
        for k in _VECTORS:
            flat[k] = np.concatenate([ep[k] for ep in self.eps])
        for split in ("obs", "next_obs"):
            keys = self.eps[0][split].keys()
            flat[split] = {k: np.concatenate([ep[split][k] for ep in self.eps])
                           for k in keys}
        self._flat = flat
        self._n_flat = int(flat["cost"].shape[0])

    def __len__(self) -> int:
        self._rebuild()
        return self._n_flat

    @property
    def ready(self) -> bool:
        return len(self) >= self.warmup_steps

    # ------------------------------------------------------------------
    def sample(self, batch_size: int) -> Dict:
        self._rebuild()
        assert self._flat is not None and self._n_flat > 0
        idx = self.rng.integers(0, self._n_flat, size=batch_size)
        out: Dict[str, object] = {}
        for k in _SCALARS:
            out[k] = self._flat[k][idx]  # type: ignore[index]
        for k in _VECTORS:
            out[k] = self._flat[k][idx]  # type: ignore[index]
        for split in ("obs", "next_obs"):
            out[split] = {k: v[idx] for k, v in self._flat[split].items()}  # type: ignore[index]
        return out

    def sample_sequences(self, n_seq: int = 16, window: int = 8):
        """Sample TBPTT windows of peer-action summaries.

        Returns ``(inputs [S*N, W, 6], targets [S*N, W, 6], h0 [S*N, H])`` where
        each (window, agent) pair is an independent sequence.
        """
        valid = [i for i, ep in enumerate(self.eps) if ep["cost"].shape[0] > window + 1]
        if not valid:
            return None
        picks = self.rng.choice(valid, size=min(n_seq, len(valid)), replace=True)
        inputs, targets, h0 = [], [], []
        for i in picks:
            ep = self.eps[i]
            T = ep["cost"].shape[0]
            start = int(self.rng.integers(0, T - window - 1))
            inputs.append(ep["peer_act"][start:start + window])          # [W,N,6]
            targets.append(ep["peer_act"][start + 1:start + window + 1])
            h0.append(ep["opp_ctx"][start])                              # [N,H]
        xs = np.stack(inputs).transpose(0, 2, 1, 3)                      # [S,N,W,6]
        ys = np.stack(targets).transpose(0, 2, 1, 3)
        hs = np.stack(h0)                                                # [S,N,H]
        S, N, W, F = xs.shape
        return (xs.reshape(S * N, W, F), ys.reshape(S * N, W, F),
                hs.reshape(S * N, -1))


__all__ = ["EpisodeBuffer"]
