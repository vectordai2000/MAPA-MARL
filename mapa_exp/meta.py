"""Simplified bi-level meta-learning layer (Algorithm 2).

What is implemented
-------------------
ξ = {θ_actor, φ_critic} is the meta-initialisation.  Each meta-iteration:

1. sample ``K`` MEC tasks from ``p(T)`` (device count / task size / mobility);
2. **inner adaptation** -- clone ξ, collect one episode of transitions on the
   task and run ``K`` first-order inner updates.  ``Δξ = mean_i(ξ_i - ξ)`` is
   the Reptile/first-order approximation of the meta-gradient (the ``β`` term of
   Eq. 18), which is what an edge UAV can actually compute;
3. **peer gradient** -- ``g_peer = ∇_ξ L_peer`` where ``L_peer`` is an
   opponent-consistency loss: the actor is asked to produce the same action when
   conditioned on the RNN's *predicted* opponent context as under the observed
   one.  This is the demo-level realisation of
   ``∇_ξ L_peer = E[∇_{π̂-n} L_inner · ∂π̂-n/∂M_ψ · ∂M_ψ/∂h · ∂h/∂ξ]``;
4. **HVP curvature compensation** -- ``H g_peer`` with
   ``H = ∇²_ξ L_inner`` (``L_inner`` = the actor's DPG loss) computed by double
   backpropagation on a fixed batch, then used in
   ``ξ ← ξ + βΔξ + η(-(I - αH) g_peer)`` (Eq. 18, descent convention);
5. **GAC** (Eq. 19) -- when ``meta_delay > 0`` the meta-direction is served from
   ``τ`` iterations ago and projected onto the current parameter manifold.

The meta-initialisation is saved and can be loaded by ``train.py`` via
``--set train.meta_init_path=...``; the task-burst experiment then measures how
fast the swarm recovers, which is the claim of Sec. V-G.
"""

from __future__ import annotations

import copy
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .agent import MAPAAgent, obs_to_tensors
from .config import Config, EnvConfig
from .env import UAVMECEnv
from .replay import EpisodeBuffer


def _flatten(params: List[torch.Tensor]) -> torch.Tensor:
    return torch.cat([p.reshape(-1) for p in params])


def _unflatten(vec: torch.Tensor, params: List[torch.Tensor]) -> None:
    offset = 0
    for p in params:
        n = p.numel()
        p.data.copy_(vec[offset:offset + n].view_as(p))
        offset += n


class MetaTrainer:
    def __init__(self, cfg: Config, device: Optional[str] = None, seed: int = 0):
        self.cfg = cfg
        self.device = device or cfg.train.device
        self.meta = MAPAAgent(cfg, device=self.device, seed=seed)
        self.inner = MAPAAgent(cfg, device=self.device, seed=seed + 1)
        self.rng = np.random.default_rng(seed)
        self.delay = max(0, int(cfg.train.meta_delay))
        self._dirs: deque = deque(maxlen=self.delay + 1)
        self.history: Dict[str, List[float]] = {k: [] for k in
                                                ("inner_reward", "delta_norm",
                                                 "peer_norm", "hvp_norm")}

    # ------------------------------------------------------------------
    def sample_task(self) -> Dict[str, float]:
        ec = self.cfg.env
        n_dev = int(self.rng.integers(max(8, ec.n_device - 6), ec.n_device + 5))
        return {
            "n_device": min(n_dev, ec.n_device_max),
            "task_scale": float(self.rng.choice([0.8, 1.0, 1.2])),
            "device_speed": float(self.rng.choice([1.0, 1.5, 2.0])),
        }

    def _make_env(self, task: Dict[str, float], seed: int) -> UAVMECEnv:
        ec: EnvConfig = copy.deepcopy(self.cfg.env)
        ec.device_speed = task["device_speed"]
        return UAVMECEnv(ec, seed=seed, n_device=int(task["n_device"]),
                         task_scale=float(task["task_scale"]))

    # ------------------------------------------------------------------
    def _collect(self, agent: MAPAAgent, env: UAVMECEnv) -> float:
        agent.buffer = EpisodeBuffer(max_episodes=4, seed=0, warmup_steps=1)
        agent.lagrangian.reset()
        obs = env.reset()
        agent.begin_episode()
        ret = 0.0
        while not env.done:
            action = agent.act(obs, explore=True)
            nobs, info = env.step(action["speed"], action["direction"], action["mode"])
            ret += -info["cost"] + agent.lagrangian.value * (
                info["jfi"] - self.cfg.env.jfi_threshold)
            agent.push(obs, action, info, nobs, env.done)
            agent.lagrangian.update(info["jfi"])
            obs = nobs
        agent.end_episode()
        return ret

    # ------------------------------------------------------------------
    def _inner_adaptation(self, tasks: List[Dict[str, float]]) -> torch.Tensor:
        """Return the mean adapted-parameter delta ``Δξ`` (Eq. 18, β term)."""
        meta_params = list(self.meta.actor.parameters()) + list(self.meta.critic.parameters())
        deltas = []
        rewards = []
        for i, task in enumerate(tasks):
            self.inner.load_state_dict(self.meta.state_dict())
            env = self._make_env(task, seed=int(self.rng.integers(1, 10_000)))
            rewards.append(self._collect(self.inner, env))
            for _ in range(self.cfg.train.meta_inner_steps):
                self.inner.update()
            inner_params = list(self.inner.actor.parameters()) + \
                list(self.inner.critic.parameters())
            deltas.append(torch.cat([(pi.detach() - pm.detach()).reshape(-1)
                                     for pi, pm in zip(inner_params, meta_params)]))
        self.history["inner_reward"].append(float(np.mean(rewards)))
        return torch.stack(deltas).mean(0)

    # ------------------------------------------------------------------
    def _peer_and_hvp(self, batch: Dict) -> Dict[str, float]:
        """Peer gradient, curvature term and the inner-loss gradient (Eq. 18-19)."""
        cfg = self.cfg
        agent = self.meta
        obs = obs_to_tensors(batch["obs"], cfg, agent.device)
        ctx = torch.as_tensor(batch["opp_ctx"], device=agent.device)
        ctx_next = torch.as_tensor(batch["opp_ctx_next"], device=agent.device)
        peer_act = torch.as_tensor(batch["peer_act"], device=agent.device)
        B, N = ctx.shape[:2]

        params = [p for p in agent.actor.parameters() if p.requires_grad]

        # --- L_inner: actor DPG loss on the task batch -----------------
        cont, logits = agent.actor(obs, ctx)
        act_now = agent.continuous_action(cont)
        k_hat = logits.argmax(-1).detach()
        q_cand = agent._candidate_q(obs, act_now, k_hat)
        q_taken = q_cand.gather(-1, k_hat.unsqueeze(-1)).squeeze(-1)
        loss_inner = -q_taken.mean()

        # --- L_peer: opponent-context consistency ---------------------
        # The RNN predicts the peers' next action summary from its hidden state;
        # feeding *that prediction* back (instead of the observed action) yields
        # the model-based opponent context, i.e. what the meta-policy would see
        # if it trusted its opponent model.  The consistency loss penalises the
        # sensitivity of the actor to that prediction error.
        with torch.no_grad():
            act_ref = agent.continuous_action(agent.actor(obs, ctx_next)[0])
        peer_hat = agent.opp.head(ctx)                     # predicted peer action
        h_pred, _ = agent.opp.step(peer_hat.reshape(-1, peer_hat.shape[-1]),
                                   ctx.reshape(-1, ctx.shape[-1]))
        ctx_pred = h_pred.reshape(B, N, -1)
        act_pred = agent.continuous_action(agent.actor(obs, ctx_pred)[0])
        loss_peer = F.mse_loss(act_pred, act_ref.detach())

        g_inner = torch.autograd.grad(loss_inner, params, retain_graph=True,
                                      create_graph=True, allow_unused=True)
        g_inner = [torch.zeros_like(p) if g is None else g for g, p in zip(g_inner, params)]
        flat_inner = _flatten(g_inner)

        g_peer = torch.autograd.grad(loss_peer, params, retain_graph=False,
                                     allow_unused=True)
        g_peer = [torch.zeros_like(p) if g is None else g.detach()
                  for g, p in zip(g_peer, params)]
        flat_peer = _flatten(g_peer)

        if cfg.train.meta_hvp_coef > 0:
            hvp = torch.autograd.grad(flat_inner, params, grad_outputs=flat_peer,
                                      allow_unused=True, retain_graph=False)
            hvp = [torch.zeros_like(p) if h is None else h.detach()
                   for h, p in zip(hvp, params)]
            flat_hvp = _flatten(hvp)
        else:
            flat_hvp = torch.zeros_like(flat_peer)

        # descent direction for the peer term: -(I - alpha H) g_peer, normalised
        # so that ``eta`` is a step size in parameter space rather than a
        # gradient-scale-dependent coefficient
        raw = -(flat_peer - cfg.train.meta_hvp_coef * flat_hvp)
        direction = raw / raw.norm().clamp_min(1e-12)
        return {"peer": flat_peer, "hvp": flat_hvp, "direction": direction,
                "loss_inner": float(loss_inner), "loss_peer": float(loss_peer)}

    # ------------------------------------------------------------------
    def meta_iteration(self, it: int) -> Dict[str, float]:
        cfg = self.cfg
        tasks = [self.sample_task() for _ in range(cfg.train.meta_tasks)]
        delta = self._inner_adaptation(tasks)

        # differentiable terms on the last task's transitions
        batch = self.inner.buffer.sample(cfg.train.batch_size)
        terms = self._peer_and_hvp(batch)

        params = list(self.meta.actor.parameters()) + list(self.meta.critic.parameters())
        actor_params = [p for p in self.meta.actor.parameters() if p.requires_grad]
        before = _flatten(params).clone()

        # the peer/HVP direction only spans the actor parameters, which are the
        # first block of ``params`` in parameter order
        act_dim = sum(p.numel() for p in actor_params)
        direction = terms["direction"]
        assert direction.numel() == act_dim
        full_dir = torch.zeros_like(before)
        full_dir[:act_dim] = direction

        # ---- GAC (Eq. 19): use a stale direction and project it ----
        if self.delay > 0 and len(self._dirs) >= self.delay:
            stale_dir, snapshot = self._dirs[-self.delay]
            dxi = before - snapshot
            denom = (stale_dir.norm() * dxi.norm()).clamp_min(1e-8)
            cos = torch.dot(stale_dir, dxi) / denom
            full_dir = stale_dir + float(cfg.train.meta_gac_eta) * cos * (
                stale_dir.norm() / dxi.norm().clamp_min(1e-8)) * dxi

        self._dirs.append((full_dir.clone(), before.clone()))

        # ---- Eq. (18) ------------------------------------------------
        update = cfg.train.meta_lr * delta + cfg.train.meta_eta * full_dir
        new = before + update
        _unflatten(new, params)

        self.history["delta_norm"].append(float(delta.norm()))
        self.history["peer_norm"].append(float(terms["peer"].norm()))
        self.history["hvp_norm"].append(float(terms["hvp"].norm()))
        return {"iter": it, "inner_reward": self.history["inner_reward"][-1],
                "delta_norm": self.history["delta_norm"][-1],
                "peer_norm": self.history["peer_norm"][-1],
                "hvp_norm": self.history["hvp_norm"][-1],
                "update_norm": float(update.norm())}

    # ------------------------------------------------------------------
    def run(self, iters: int, log_every: int = 1, verbose: bool = True) -> List[Dict]:
        logs = []
        for it in range(1, iters + 1):
            log = self.meta_iteration(it)
            logs.append(log)
            if verbose and it % log_every == 0:
                print(f"[meta {it:3d}] innerR {log['inner_reward']:7.2f} | "
                      f"|dxi| {log['delta_norm']:.4f} | |g| {log['peer_norm']:.4f} | "
                      f"|Hv| {log['hvp_norm']:.4f} | |upd| {log['update_norm']:.4f}")
        return logs

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"meta_init": self.meta.state_dict(),
                    "history": self.history}, path)


__all__ = ["MetaTrainer"]
