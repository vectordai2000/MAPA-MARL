"""MAPPO baselines with the three discrete-action treatments of Sec. V-B.

The paper compares against MAPPO in three flavours:

* ``native``         -- native categorical sampling (its gradient stream has a
  completely different scale from the continuous Gaussian stream),
* ``native_clipped`` -- the same, plus independent optimisers and aggressive
  gradient clipping (a brute-force engineering patch),
* ``stgs``           -- a straight-through Gumbel-Softmax relaxation with the
  temperature schedule ``tau = max(0.1, 5 exp(-0.005 step))``.

All three share the *identical* attention encoder with MAPA-MARL (the paper asks
for exactly that), use a centralised state-value critic ``V(s)``, GAE(lambda) and
a clipped PPO surrogate.  The only difference between variants is how the
categorical head is differentiated.

Interface: this class mirrors :class:`mapa_exp.agent.MAPAAgent` (``begin_episode``,
``act``, ``push``, ``end_episode``, ``update``, ``reset_hidden``,
``advance_hidden``) so that ``mapa_exp.train`` can drive either agent unchanged.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .agent import obs_to_tensors
from .config import Config
from .env import UAVMECEnv
from .estimators import anneal_temperature
from .lagrangian import PIDLagrangian
from .networks import CentralValue, DualHeadActor, OpponentModel


class MAPPOAgent:
    def __init__(self, cfg: Config, device: Optional[str] = None, seed: int = 0):
        self.cfg = cfg
        self.tc = cfg.train
        self.ec = cfg.env
        self.variant = self.tc.estimator if self.tc.estimator in (
            "native", "native_clipped", "stgs") else "native"
        self.device = torch.device(device or self.tc.device)
        torch.manual_seed(seed)

        self.n_agent = cfg.env.n_uav
        self.h_dim = self.tc.rnn_hidden

        self.actor = DualHeadActor(
            d_model=self.tc.d_model, n_heads=self.tc.n_heads,
            n_layers=self.tc.n_layers, dim_self=UAVMECEnv.DIM_SELF,
            dim_peer=UAVMECEnv.DIM_PEER, dim_dev=UAVMECEnv.DIM_DEV,
            rnn_hidden=self.tc.rnn_hidden if self.tc.use_rnn else 0,
            cont_dim=2,                       # [speed_raw, yaw_raw]
        ).to(self.device)
        self.log_std = nn.Parameter(torch.full((2,), -0.7, device=self.device))
        self.value = CentralValue(
            d_model=self.tc.critic_d_model, n_heads=self.tc.n_heads,
            n_layers=self.tc.critic_n_layers,
            dim_agent=UAVMECEnv.DIM_AGENT, dim_dev=UAVMECEnv.DIM_DEV_HIDDEN,
        ).to(self.device)
        self.opp = OpponentModel(dim_in=UAVMECEnv.DIM_PEER_ACT,
                                 hidden=self.tc.rnn_hidden,
                                 dim_out=UAVMECEnv.DIM_PEER_ACT).to(self.device)

        act_params = [p for n, p in self.actor.named_parameters()
                      if not n.startswith("head_disc")] + [self.log_std]
        disc_params = list(self.actor.head_disc.parameters())
        self.opt_actor = torch.optim.Adam(
            [{"params": act_params, "lr": self.tc.lr_actor},
             {"params": disc_params, "lr": self.tc.lr_actor_disc}])
        self.opt_value = torch.optim.Adam(self.value.parameters(), lr=self.tc.lr_critic)
        self.opt_opp = torch.optim.Adam(self.opp.parameters(), lr=self.tc.lr_opponent)

        self.lagrangian = PIDLagrangian(
            jfi_threshold=self.ec.jfi_threshold, k_p=self.tc.k_p,
            k_i=self.tc.k_i, k_d=self.tc.k_d, lambda_init=self.tc.lambda_init,
            lambda_max=self.tc.lambda_max, ema_beta=self.tc.ema_beta,
            ema_delta=self.tc.ema_delta, ema_kappa=self.tc.ema_kappa,
            use_pid=self.tc.use_pid)

        self._h = torch.zeros(1, self.n_agent, self.h_dim, device=self.device)
        self._ep: List[Dict] = []
        self._pending: Optional[Dict] = None
        self._buffer: List[Dict] = []
        self.n_updates = 0
        self.ppo = {"epochs": 4, "minibatch": 128, "clip": 0.2,
                    "value_coef": 0.5, "entropy_coef": 0.01,
                    "gae_lambda": 0.95, "max_steps": 512}

    # ------------------------------------------------------------------
    # policy
    # ------------------------------------------------------------------
    def _dist(self, obs, ctx):
        mu, logits = self.actor(obs, ctx)
        std = self.log_std.exp().clamp(1e-3, 2.0)
        return mu, std, logits

    @staticmethod
    def _decode(z: torch.Tensor):
        speed = 0.5 * (1.0 + torch.tanh(z[..., 0]))
        yaw = z[..., 1]
        direction = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)
        return speed, direction, yaw

    def _logp(self, z, mu, std):
        normal = torch.distributions.Normal(mu, std)
        lp = normal.log_prob(z).sum(-1)
        # tanh correction on the speed dimension
        lp = lp - torch.log(0.5 * (1.0 - torch.tanh(z[..., 0]) ** 2) + 1e-6)
        return lp

    @torch.no_grad()
    def act(self, obs: Dict[str, np.ndarray], explore: bool = True) -> Dict[str, np.ndarray]:
        o = obs_to_tensors(obs, self.cfg, self.device)
        mu, std, logits = self._dist(o, self._h)
        if explore:
            z = mu + std * torch.randn_like(mu)
        else:
            z = mu
        speed, direction, yaw = self._decode(z)
        probs = torch.softmax(logits, dim=-1)
        if explore:
            mode = torch.multinomial(probs.reshape(-1, 3), 1).reshape(self.n_agent)
            logp_mode = F.log_softmax(logits, -1).reshape(-1, 3).gather(
                1, mode.reshape(-1, 1)).reshape(1, self.n_agent)
            logp = (self._logp(z, mu, std) + logp_mode[0])[0]          # [N]
            value = self.value(o)[0]
        else:
            mode = logits.argmax(-1).reshape(-1)
            logp = torch.zeros(self.n_agent, device=self.device)
            value = torch.zeros(self.n_agent, device=self.device)

        self._pending = {"z": z[0], "logp": logp, "value": value,
                         "ctx": self._h[0].clone()}
        return {"speed": speed[0].cpu().numpy(),
                "direction": direction[0].cpu().numpy(),
                "mode": mode.cpu().numpy()}

    # ------------------------------------------------------------------
    # rollout bookkeeping
    # ------------------------------------------------------------------
    def begin_episode(self) -> None:
        self._h = torch.zeros(1, self.n_agent, self.h_dim, device=self.device)
        self._ep = []

    def push(self, obs, action, info, next_obs, done) -> None:
        if self._pending is None:
            self.advance_hidden(next_obs["peer_act"])
            return
        reward = -info["cost"]
        if self.tc.use_pid:
            reward += self.lagrangian.value * (info["jfi"] - self.ec.jfi_threshold)
        self._ep.append({
            "obs": {k: np.asarray(v, dtype=np.float32) for k, v in obs.items()},
            "z": self._pending["z"].cpu().numpy(),
            "mode": np.asarray(action["mode"], dtype=np.int64),
            "logp": self._pending["logp"].cpu().numpy(),
            "value": self._pending["value"].cpu().numpy(),
            "ctx": self._pending["ctx"].cpu().numpy(),
            "reward": float(reward),
            "done": float(done),
        })
        self._pending = None
        self.advance_hidden(next_obs["peer_act"])

    @torch.no_grad()
    def advance_hidden(self, peer_act: np.ndarray) -> torch.Tensor:
        peer = torch.as_tensor(np.asarray(peer_act, dtype=np.float32), device=self.device)
        h_next, _ = self.opp.step(peer, self._h.reshape(-1, self.h_dim))
        self._h = h_next.reshape(1, self.n_agent, self.h_dim)
        return self._h[0].clone()

    def reset_hidden(self) -> None:
        self._h = torch.zeros(1, self.n_agent, self.h_dim, device=self.device)

    def end_episode(self):
        if not self._ep:
            return None
        ep = {
            "obs": {k: np.stack([s["obs"][k] for s in self._ep])
                    for k in self._ep[0]["obs"]},
            "z": np.stack([s["z"] for s in self._ep]),
            "mode": np.stack([s["mode"] for s in self._ep]),
            "logp": np.stack([s["logp"] for s in self._ep]),
            "value": np.stack([s["value"] for s in self._ep]),
            "ctx": np.stack([s["ctx"] for s in self._ep]),
            "reward": np.asarray([s["reward"] for s in self._ep]),
            "done": np.asarray([s["done"] for s in self._ep]),
        }
        self._buffer.append(ep)
        self._ep = []
        total = sum(e["reward"].shape[0] for e in self._buffer)
        if total >= self.ppo["max_steps"]:
            result = self._ppo_update()
            self._buffer = []
            return result
        return None

    def update(self) -> Optional[Dict[str, float]]:
        return None                              # MAPPO is on-policy: see end_episode

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------
    def _flatten(self):
        eps = self._buffer
        obs = {k: np.concatenate([e["obs"][k] for e in eps]) for k in eps[0]["obs"]}
        data = {k: np.concatenate([e[k] for e in eps])
                for k in ("z", "mode", "logp", "value", "ctx", "reward", "done")}
        return obs, data

    def _gae(self, data):
        """Advantages per agent from the stored values (single trajectory order)."""
        rewards = torch.as_tensor(data["reward"], dtype=torch.float32,
                                  device=self.device)          # [T]
        values = torch.as_tensor(data["value"], dtype=torch.float32,
                                 device=self.device)           # [T,N]
        dones = torch.as_tensor(data["done"], dtype=torch.float32, device=self.device)
        T = rewards.shape[0]
        adv = torch.zeros_like(values)
        last = torch.zeros_like(values[0])
        lam, gamma = self.ppo["gae_lambda"], self.tc.gamma
        for t in reversed(range(T)):
            next_v = values[t + 1] if t + 1 < T else torch.zeros_like(values[0])
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_v * nonterminal - values[t]
            last = delta + gamma * lam * nonterminal * last
            adv[t] = last
        returns = adv + values
        return adv, returns

    def _ppo_update(self) -> Dict[str, float]:
        obs_np, data = self._flatten()
        obs = obs_to_tensors(obs_np, self.cfg, self.device)
        z = torch.as_tensor(data["z"], device=self.device)
        mode = torch.as_tensor(data["mode"], device=self.device)
        logp_old = torch.as_tensor(data["logp"], device=self.device)
        ctx_all = torch.as_tensor(data["ctx"], device=self.device)
        adv, returns = self._gae(data)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        n = z.shape[0]
        tau_start, tau_end = self.tc.stgs_tau_start, self.tc.stgs_tau_end
        last = {}
        for epoch in range(self.ppo["epochs"]):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, self.ppo["minibatch"]):
                idx = perm[start:start + self.ppo["minibatch"]]
                mb_obs = {k: v[idx] for k, v in obs.items()}
                mb_ctx = ctx_all[idx]
                mu, std, logits = self._dist(mb_obs, mb_ctx)
                logp = self._logp(z[idx], mu, std) + F.log_softmax(
                    logits, -1).gather(-1, mode[idx].unsqueeze(-1)).squeeze(-1)
                ratio = (logp - logp_old[idx]).exp()
                a = adv[idx]
                surr = torch.min(ratio * a, ratio.clamp(1 - self.ppo["clip"],
                                                        1 + self.ppo["clip"]) * a)
                if self.variant == "stgs":
                    tau = anneal_temperature(self.n_updates, 2000, tau_start, tau_end)
                    g = F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)
                    loss_disc = -(g * a.unsqueeze(-1)).sum(-1).mean()
                    loss_pi = -surr.mean() + loss_disc
                else:
                    loss_pi = -surr.mean()
                entropy = -(torch.softmax(logits, -1)
                            * torch.log_softmax(logits, -1)).sum(-1).mean()
                v = self.value(mb_obs)
                loss_v = F.mse_loss(v, returns[idx])
                loss = loss_pi + self.ppo["value_coef"] * loss_v - \
                    self.ppo["entropy_coef"] * entropy

                self.opt_actor.zero_grad(set_to_none=True)
                self.opt_value.zero_grad(set_to_none=True)
                loss.backward()
                if self.variant == "native_clipped":
                    # aggressive per-head clipping: the paper's brute-force patch
                    nn.utils.clip_grad_norm_(list(self.actor.head_disc.parameters()),
                                             0.1)
                    nn.utils.clip_grad_norm_(list(self.actor.head_cont.parameters()),
                                             0.1)
                else:
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.tc.grad_clip)
                self.opt_actor.step()
                nn.utils.clip_grad_norm_(self.value.parameters(), self.tc.grad_clip)
                self.opt_value.step()
                last = {"pi_loss": float(loss_pi), "v_loss": float(loss_v),
                        "entropy": float(entropy)}
        self.n_updates += 1
        return last

    # ------------------------------------------------------------------
    def state_dict(self) -> Dict:
        return {"actor": self.actor.state_dict(), "critic": self.value.state_dict(),
                "opp": self.opp.state_dict(), "log_std": self.log_std.detach().cpu()}

    def load_state_dict(self, state: Dict) -> None:
        self.actor.load_state_dict(state["actor"], strict=False)
        self.value.load_state_dict(state["critic"])
        if "opp" in state:
            self.opp.load_state_dict(state["opp"])
        if "log_std" in state:
            with torch.no_grad():
                self.log_std.copy_(state["log_std"].to(self.device))

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        if "meta_init" in state:
            state = state["meta_init"]
        self.load_state_dict(state)


__all__ = ["MAPPOAgent"]
