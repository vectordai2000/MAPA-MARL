"""The MAPA-MARL agent: everything that is *not* the environment.

Composition (Sec. IV):

* Dual-head actor with shared attention encoder (Sec. IV-A).
* Centralised masked-attention critic (Sec. IV-B) with the ZGR discrete head.
* RNN implicit opponent model (Sec. IV-C) whose hidden state is the opponent
  context used by the actor, trained with truncated BPTT.
* Adaptive EMA-filtered PID-Lagrangian multiplier for the fairness CMDP
  (Sec. III-E).

The update follows Algorithm 1::
    critic  : y = r_cost - lambda * r_fair + gamma Q'(s', a')        (16)
    actor   : continuous head by DPG on Q, discrete head by ZGR proxy (13)
    targets : soft update
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .config import Config
from .env import UAVMECEnv
from .estimators import (anneal_temperature, categorical_entropy, stgs_one_hot,
                         zgr_surrogate)
from .lagrangian import PIDLagrangian
from .mask import log_mask_bias, mask_from_distance
from .networks import CentralCritic, DualHeadActor, OpponentModel
from .replay import EpisodeBuffer

_TENSOR_KEYS = ("self", "peer", "dev", "agent_hd", "dev_hd", "dist_uu", "dist_ud",
                "peer_act", "peer_mask", "dev_mask")


# ---------------------------------------------------------------------------
# observation -> tensors (also builds the spatial attention bias)
# ---------------------------------------------------------------------------
def obs_to_tensors(obs: Dict[str, np.ndarray], cfg: Config,
                   device: torch.device) -> Dict[str, torch.Tensor]:
    """Convert an env observation to tensors of shape ``[B, N, ...]``.

    The spatial mask ``M`` of Eq. (15) is rebuilt here from the raw distances so
    that rollout and training always use exactly the same operator, and so that
    the "w/o Sparse-Mask" ablation only changes the *network*, not the world.
    """
    tc = cfg.train
    batched = np.asarray(obs["self"]).ndim == 3

    # Pack every field into one contiguous buffer: a single host->device copy
    # instead of one per key (the observation has 10 fields, and on a GPU the
    # per-copy launch overhead dominated the step time).
    arrays, shapes = [], []
    for key in _TENSOR_KEYS:
        arr = np.asarray(obs[key], dtype=np.float32)
        if not batched:
            arr = arr[None]
        arrays.append(arr.reshape(-1))
        shapes.append(arr.shape)
    packed = np.concatenate(arrays) if len(arrays) > 1 else arrays[0]
    flat = torch.as_tensor(packed, device=device)

    out: Dict[str, torch.Tensor] = {}
    offset = 0
    for key, shape in zip(_TENSOR_KEYS, shapes):
        n = int(np.prod(shape))
        out[key] = flat[offset:offset + n].view(shape)
        offset += n

    B, N = out["self"].shape[:2]
    eye_idx = torch.tensor(
        [[j for j in range(N) if j != i] for i in range(N)],
        dtype=torch.long, device=device,
    ).unsqueeze(0).expand(B, -1, -1)

    if tc.use_mask:
        m_ud = mask_from_distance(out["dist_ud"], tc.mask_sigma_d, tc.mask_gamma,
                                  tc.mask_tau, tc.mask_eps_th)
        m_uu = mask_from_distance(out["dist_uu"], tc.mask_sigma_d, tc.mask_gamma,
                                  tc.mask_tau, tc.mask_eps_th)
    else:  # ablation: dense (fully connected) attention with uniform weights
        m_ud = torch.ones_like(out["dist_ud"])
        m_uu = torch.ones_like(out["dist_uu"])

    out["bias_ud"] = log_mask_bias(m_ud)                       # [B,N,M]
    out["bias_uu"] = log_mask_bias(m_uu)                       # [B,N,N]
    peer_bias = out["bias_uu"].gather(2, eye_idx)              # [B,N,N-1]

    zeros = torch.zeros(B, N, 1, device=device)
    ones = torch.ones(B, N, 1, device=device)
    out["attn_bias"] = torch.cat([zeros, peer_bias, out["bias_ud"]], dim=-1)
    out["attn_pad"] = torch.cat([ones, out["peer_mask"], out["dev_mask"]], dim=-1)
    out["dev_present"] = out["dev_mask"][:, 0]                 # [B,M]
    return out


# ---------------------------------------------------------------------------
# agent
# ---------------------------------------------------------------------------
class MAPAAgent:
    def __init__(self, cfg: Config, device: Optional[str] = None, seed: int = 0):
        self.cfg = cfg
        self.tc = cfg.train
        self.ec = cfg.env
        if device is None:
            device = cfg.train.device
        self.device = torch.device(
            device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
        torch.manual_seed(seed)

        self.n_agent = cfg.env.n_uav
        self.m_max = cfg.env.n_device_max
        self.h_dim = self.tc.rnn_hidden

        self.actor = DualHeadActor(
            d_model=self.tc.d_model, n_heads=self.tc.n_heads,
            n_layers=self.tc.n_layers,
            dim_self=UAVMECEnv.DIM_SELF, dim_peer=UAVMECEnv.DIM_PEER,
            dim_dev=UAVMECEnv.DIM_DEV,
            rnn_hidden=self.tc.rnn_hidden if self.tc.use_rnn else 0,
        ).to(self.device)
        self.critic = CentralCritic(
            d_model=self.tc.critic_d_model, n_heads=self.tc.n_heads,
            n_layers=self.tc.critic_n_layers,
            dim_agent=UAVMECEnv.DIM_AGENT, dim_dev=UAVMECEnv.DIM_DEV_HIDDEN,
        ).to(self.device)
        self.opp = OpponentModel(
            dim_in=UAVMECEnv.DIM_PEER_ACT, hidden=self.tc.rnn_hidden,
            dim_out=UAVMECEnv.DIM_PEER_ACT,
        ).to(self.device)

        self.actor_targ = self._clone(self.actor)
        self.critic_targ = self._clone(self.critic)

        encoder = [p for n, p in self.actor.named_parameters()
                   if not n.startswith("head_")]
        self.opt_actor = torch.optim.Adam([
            {"params": encoder, "lr": self.tc.lr_actor},
            {"params": list(self.actor.head_cont.parameters()), "lr": self.tc.lr_actor},
            {"params": list(self.actor.head_disc.parameters()), "lr": self.tc.lr_actor_disc},
        ])
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=self.tc.lr_critic)
        self.opt_opp = torch.optim.Adam(self.opp.parameters(), lr=self.tc.lr_opponent)

        self.lagrangian = PIDLagrangian(
            jfi_threshold=self.ec.jfi_threshold,
            k_p=self.tc.k_p, k_i=self.tc.k_i, k_d=self.tc.k_d,
            lambda_init=self.tc.lambda_init, lambda_max=self.tc.lambda_max,
            ema_beta=self.tc.ema_beta, ema_delta=self.tc.ema_delta,
            ema_kappa=self.tc.ema_kappa, use_pid=self.tc.use_pid,
        )
        self.buffer = EpisodeBuffer(max_episodes=self.tc.buffer_episodes,
                                    seed=seed, warmup_steps=self.tc.warmup_steps)
        self._h = torch.zeros(1, self.n_agent, self.h_dim, device=self.device)
        self.n_updates = 0

    # ------------------------------------------------------------------
    @staticmethod
    def _clone(module: nn.Module) -> nn.Module:
        import copy
        clone = copy.deepcopy(module)
        for p in clone.parameters():
            p.requires_grad_(False)
        return clone

    # ------------------------------------------------------------------
    # rollout
    # ------------------------------------------------------------------
    def begin_episode(self) -> None:
        self._h = torch.zeros(1, self.n_agent, self.h_dim, device=self.device)
        self.buffer.begin_episode()

    @torch.no_grad()
    def act(self, obs: Dict[str, np.ndarray], explore: bool = True) -> Dict[str, np.ndarray]:
        o = obs_to_tensors(obs, self.cfg, self.device)
        cont, logits = self.actor(o, self._h)
        act = self.continuous_action(cont)                     # [1,N,3] in env units
        speed, direction = act[..., 0], act[..., 1:]

        if explore:
            angle = torch.atan2(direction[..., 1], direction[..., 0])
            angle = angle + torch.randn_like(angle) * self.tc.explore_dir_std
            direction = torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)
            speed = (speed + torch.randn_like(speed) * self.tc.explore_speed_std).clamp(0.0, 1.0)
            probs = torch.softmax(logits / self.tc.explore_temp, dim=-1)
            mode = torch.multinomial(probs.reshape(-1, 3), 1).reshape(self.n_agent)
        else:
            mode = logits.argmax(-1).reshape(self.n_agent)

        # one device->host copy for the whole action instead of three
        packed = torch.cat([speed[0][:, None], direction[0],
                            mode.reshape(-1, 1).float()], dim=-1).cpu().numpy()
        return {"speed": packed[:, 0], "direction": packed[:, 1:3],
                "mode": packed[:, 3].astype(np.int64)}

    @staticmethod
    def continuous_action(cont: torch.Tensor) -> torch.Tensor:
        """Map the continuous head output to the physical action ``[v, dx, dy]``.

        The head emits a raw speed scalar and a raw 2-D direction; the critic is
        always fed the *processed* action so that rollout, target evaluation and
        actor improvement all live in the same action manifold.
        """
        speed = 0.5 * (torch.tanh(cont[..., 0]) + 1.0)
        direction = cont[..., 1:]
        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-6)
        return torch.cat([speed.unsqueeze(-1), direction], dim=-1)

    def push(self, obs: Dict[str, np.ndarray], action: Dict[str, np.ndarray],
             info: Dict[str, float], next_obs: Dict[str, np.ndarray],
             done: bool) -> None:
        """Store one transition and advance the opponent-model hidden state."""
        ctx_now = self._h[0].clone()
        ctx_next = self.advance_hidden(next_obs["peer_act"])

        act_cont = np.concatenate([action["speed"][:, None],
                                   action["direction"]], axis=1)
        self.buffer.push(obs, act_cont, action["mode"], info, next_obs, done,
                         ctx_now.cpu().numpy(), ctx_next.cpu().numpy())

    @torch.no_grad()
    def advance_hidden(self, peer_act: np.ndarray) -> torch.Tensor:
        """Feed the observed peer action summary into the GRU (Sec. IV-C)."""
        peer = torch.as_tensor(np.asarray(peer_act, dtype=np.float32),
                               device=self.device)
        h_next, _ = self.opp.step(peer, self._h.reshape(-1, self.h_dim))
        self._h = h_next.reshape(1, self.n_agent, self.h_dim)
        return self._h[0].clone()

    def reset_hidden(self) -> None:
        self._h = torch.zeros(1, self.n_agent, self.h_dim, device=self.device)

    def end_episode(self) -> None:
        self.buffer.end_episode()

    # ------------------------------------------------------------------
    # learning
    # ------------------------------------------------------------------
    def update(self) -> Optional[Dict[str, float]]:
        if not self.buffer.ready:
            return None
        tc = self.tc
        batch = self.buffer.sample(tc.batch_size)
        obs = obs_to_tensors(batch["obs"], self.cfg, self.device)
        nobs = obs_to_tensors(batch["next_obs"], self.cfg, self.device)
        ctx = torch.as_tensor(batch["opp_ctx"], device=self.device)
        ctx_next = torch.as_tensor(batch["opp_ctx_next"], device=self.device)
        cost = torch.as_tensor(batch["cost"], dtype=torch.float32, device=self.device)
        jfi = torch.as_tensor(batch["jfi"], dtype=torch.float32, device=self.device)
        safety = torch.as_tensor(batch["safety"], dtype=torch.float32, device=self.device)
        done = torch.as_tensor(batch["done"], dtype=torch.float32, device=self.device)
        act_cont = torch.as_tensor(batch["act_cont"], dtype=torch.float32, device=self.device)
        act_mode = torch.as_tensor(batch["act_mode"], dtype=torch.long, device=self.device)
        act_onehot = F.one_hot(act_mode, 3).float()

        # ---- Lagrangian reward, Eq. (16) ----------------------------
        lam = self.lagrangian.value
        reward = (-(cost + self.ec.safety_weight * safety)
                  + lam * (jfi - self.ec.jfi_threshold))[:, None]      # [B,1]

        # ---- critic -------------------------------------------------
        with torch.no_grad():
            cont_t, logits_t = self.actor_targ(nobs, ctx_next)
            act_t = self.continuous_action(cont_t)
            k_t = logits_t.argmax(-1)
            q_next = self.critic_targ(nobs, act_t, F.one_hot(k_t, 3).float())
            target = reward + tc.gamma * (1.0 - done)[:, None] * q_next
        q = self.critic(obs, act_cont, act_onehot)
        loss_c = F.mse_loss(q, target)
        self.opt_critic.zero_grad(set_to_none=True)
        loss_c.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), tc.grad_clip)
        self.opt_critic.step()

        # ---- actor --------------------------------------------------
        cont, logits = self.actor(obs, ctx)
        act_now = self.continuous_action(cont)          # same manifold as data
        act_now.retain_grad()
        logits.retain_grad()

        if tc.estimator == "zgr":
            probs = torch.softmax(logits / tc.zgr_temperature, dim=-1)
            k_hat = probs.argmax(-1).detach()
            q_cand = self._candidate_q(obs, act_now, k_hat)        # [B,N,3]
            loss_disc = (zgr_surrogate(logits, q_cand, tc.zgr_temperature)
                         - tc.entropy_coef * categorical_entropy(
                             logits, tc.zgr_temperature).mean())
            q_taken = q_cand.gather(-1, k_hat.unsqueeze(-1)).squeeze(-1)
            loss_cont = -q_taken.mean()
        else:  # ST-GS baseline / "w/o ZGR" ablation
            total = max(1, tc.episodes * self.cfg.env.n_slot)
            tau = anneal_temperature(self.n_updates, total,
                                     tc.stgs_tau_start, tc.stgs_tau_end)
            k_soft = stgs_one_hot(logits, tau)
            q_taken = self.critic(obs, act_now, k_soft)
            loss_cont = -q_taken.mean()
            loss_disc = torch.zeros((), device=self.device)

        loss_a = loss_cont + loss_disc
        self.opt_actor.zero_grad(set_to_none=True)
        loss_a.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), tc.grad_clip)
        self.opt_actor.step()

        # ---- diagnostics (Fig. 6: gradient L2 norms) ----------------
        g_disc = float(logits.grad.norm(dim=-1).mean()) if logits.grad is not None else 0.0
        g_cont = float(act_now.grad.norm(dim=-1).mean()) if act_now.grad is not None else 0.0

        # ---- opponent model (TBPTT) ---------------------------------
        opp_loss = self._update_opponent() if tc.use_rnn else None

        self._soft_update(self.actor, self.actor_targ)
        self._soft_update(self.critic, self.critic_targ)
        self.n_updates += 1

        return {"critic_loss": float(loss_c), "actor_loss": float(loss_cont),
                "grad_disc": g_disc, "grad_cont": g_cont,
                "opp_loss": opp_loss if opp_loss is not None else float("nan"),
                "lambda": lam}

    # ------------------------------------------------------------------
    def _candidate_q(self, obs: Dict[str, torch.Tensor], act_cont: torch.Tensor,
                     k_hat: torch.Tensor) -> torch.Tensor:
        """Evaluate Q(s, a_n = k) for every agent and every k in one critic pass.

        The forward pass stages ``N * 3`` copies of the batch: row ``(b, n, c)``
        is the joint action where only agent ``n`` uses candidate ``c``.  This is
        the vectorised version of Eq. (13)'s candidate enumeration.
        """
        B, N = act_cont.shape[:2]
        rep = N * 3
        obs_rep = {k: v.repeat_interleave(rep, dim=0) for k, v in obs.items()}
        act_rep = act_cont.repeat_interleave(rep, dim=0)            # [B*N*3, N, 3]
        onehot = F.one_hot(k_hat, 3).float().repeat_interleave(rep, dim=0)
        rows = torch.arange(B * rep, device=act_cont.device)
        n_of_row = (rows // 3) % N
        c_of_row = rows % 3
        onehot[rows, n_of_row] = F.one_hot(c_of_row, 3).float()
        q_all = self.critic(obs_rep, act_rep, onehot)               # [B*N*3, N]
        return q_all[rows, n_of_row].view(B, N, 3)

    def _update_opponent(self) -> Optional[float]:
        seq = self.buffer.sample_sequences(n_seq=16, window=self.tc.tbptt)
        if seq is None:
            return None
        xs, ys, h0 = seq
        xs_t = torch.as_tensor(xs, device=self.device)
        ys_t = torch.as_tensor(ys, device=self.device)
        h0_t = torch.as_tensor(h0, device=self.device)
        pred, _ = self.opp(xs_t, h0_t)
        loss = F.mse_loss(pred, ys_t)
        self.opt_opp.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.opp.parameters(), self.tc.grad_clip)
        self.opt_opp.step()
        return float(loss)

    @torch.no_grad()
    def _soft_update(self, src: nn.Module, tgt: nn.Module) -> None:
        tau = self.tc.tau
        for p, pt in zip(src.parameters(), tgt.parameters()):
            pt.mul_(1.0 - tau).add_(tau * p)

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def state_dict(self) -> Dict[str, Dict]:
        return {"actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "opp": self.opp.state_dict()}

    def load_state_dict(self, state: Dict[str, Dict]) -> None:
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        if "opp" in state:
            self.opp.load_state_dict(state["opp"])
        self.actor_targ.load_state_dict(self.actor.state_dict())
        self.critic_targ.load_state_dict(self.critic.state_dict())

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        if "meta_init" in state:            # file written by ``meta.py``
            state = state["meta_init"]
        self.load_state_dict(state)


__all__ = ["MAPAAgent", "obs_to_tensors"]
