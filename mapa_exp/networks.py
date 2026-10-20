"""Networks for the MAPA-MARL demo.

Two extra pieces relative to a textbook MADDPG implementation:

* :class:`DualHeadActor` -- a shared attention encoder (inspired by LLM
  self-attention, Sec. IV-A) with two heads: a continuous flight head trained
  with DPG and a categorical offloading head trained with the ZGR proxy.
* :class:`CentralCritic` -- a masked-attention critic Q_phi(s~, a) whose
  attention logits are biased by ``log(M + delta)``, i.e. the sparsity-induced
  topological mask (Sec. IV-B).
* :class:`OpponentModel` -- a GRU that infers neighbours' policies from their
  observed actions and supplies the meta-policy with an opponent context
  (Sec. IV-C).

Parameter counts are ~10^2 smaller than the ~4.5 M reported in Table V; the
demo targets clarity, not inference latency.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# building blocks
# ---------------------------------------------------------------------------
def mlp(sizes, activation=nn.GELU, output_activation=None, gain=1.0):
    layers = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        lin = nn.Linear(a, b)
        nn.init.orthogonal_(lin.weight, gain=gain if i == len(sizes) - 2 else 1.0)
        nn.init.zeros_(lin.bias)
        layers.append(lin)
        if i < len(sizes) - 2:
            layers.append(activation())
    if output_activation is not None:
        layers.append(output_activation())
    return nn.Sequential(*layers)


class MaskedAttention(nn.Module):
    """Multi-head attention with an additive spatial bias and key padding.

    ``bias`` is added to the attention logits (it carries ``log M``), ``pad`` is
    a {0,1} mask where 0 marks padded keys.  Both are broadcast over heads.
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.dk = d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)

    def forward(self, q: torch.Tensor, kv: torch.Tensor,
                bias: Optional[torch.Tensor] = None,
                pad: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, Lq, D = q.shape
        Lk = kv.shape[1]
        qh = self.q(q).view(B, Lq, self.h, self.dk).transpose(1, 2)
        kh = self.k(kv).view(B, Lk, self.h, self.dk).transpose(1, 2)
        vh = self.v(kv).view(B, Lk, self.h, self.dk).transpose(1, 2)
        scores = torch.matmul(qh, kh.transpose(-1, -2)) / math.sqrt(self.dk)

        if bias is not None:
            if bias.dim() == 2:                       # [B, Lk] -> [B, 1, 1, Lk]
                bias = bias[:, None, None, :]
            elif bias.dim() == 3:                     # [B, Lq, Lk]
                bias = bias[:, None]
            scores = scores + bias.clamp(min=-12.0)
        if pad is not None:
            scores = scores.masked_fill(pad[:, None, None, :] < 0.5, -1e9)

        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, vh).transpose(1, 2).reshape(B, Lq, D)
        return self.o(out)


class Block(nn.Module):
    """Pre-norm transformer block (attention + feed-forward)."""

    def __init__(self, d_model: int, n_heads: int, ratio: int = 2, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MaskedAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ratio * d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(ratio * d_model, d_model),
        )

    def forward(self, x, bias=None, pad=None):
        h = self.ln1(x)
        x = x + self.attn(h, h, bias, pad)
        return x + self.ff(self.ln2(x))


# ---------------------------------------------------------------------------
# dual-head actor
# ---------------------------------------------------------------------------
class DualHeadActor(nn.Module):
    """Shared encoder + continuous (DPG) and discrete (ZGR) heads.

    Continuous output ``[speed_raw, dir_x, dir_y]``; discrete output are the
    three logits of ``k in {local, edge, cloud}`` (Eq. 12).
    """

    def __init__(self, d_model: int = 64, n_heads: int = 4, n_layers: int = 2,
                 dim_self: int = 6, dim_peer: int = 5, dim_dev: int = 6,
                 rnn_hidden: int = 32, init_dir=(1.0, 0.0), cont_dim: int = 3):
        super().__init__()
        self.cont_dim = cont_dim
        self.emb_self = nn.Linear(dim_self, d_model)
        self.emb_peer = nn.Linear(dim_peer, d_model)
        self.emb_dev = nn.Linear(dim_dev, d_model)
        self.type_emb = nn.Parameter(torch.zeros(3, d_model))
        nn.init.normal_(self.type_emb, std=0.02)

        self.blocks = nn.ModuleList([Block(d_model, n_heads) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)

        self.has_opponent = rnn_hidden > 0
        if self.has_opponent:
            self.opp_proj = nn.Sequential(
                nn.Linear(rnn_hidden, d_model), nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            trunk_in = 2 * d_model
        else:
            trunk_in = d_model
        self.trunk = mlp([trunk_in, d_model, d_model], gain=1.0)

        self.head_cont = nn.Linear(d_model, cont_dim)
        nn.init.orthogonal_(self.head_cont.weight, gain=0.01)
        with torch.no_grad():
            if cont_dim == 3:                     # [speed, dir_x, dir_y] (MAPA-MARL)
                self.head_cont.bias.copy_(torch.tensor([0.0, init_dir[0], init_dir[1]]))
            else:                                 # [speed, yaw] (MAPPO Gaussian policy)
                self.head_cont.bias.zero_()
        self.head_disc = nn.Linear(d_model, 3)
        nn.init.orthogonal_(self.head_disc.weight, gain=0.01)
        nn.init.zeros_(self.head_disc.bias)

    def forward(self, obs, opp_ctx=None):
        """obs entries keep the ``[B, N, ...]`` layout of the environment."""
        B, N = obs["self"].shape[:2]
        s = self.emb_self(obs["self"]) + self.type_emb[0]
        p = self.emb_peer(obs["peer"]) + self.type_emb[1]
        d = self.emb_dev(obs["dev"]) + self.type_emb[2]
        tokens = torch.cat([s[:, :, None], p, d], dim=2)              # [B,N,T,D]

        bias = obs["attn_bias"]                                       # [B,N,T]
        pad = obs["attn_pad"]                                         # [B,N,T]
        h = tokens.reshape(B * N, -1, tokens.shape[-1])
        bias = bias.reshape(B * N, -1)
        pad = pad.reshape(B * N, -1)
        for blk in self.blocks:
            h = blk(h, bias, pad)
        h = self.norm(h)[:, 0].reshape(B, N, -1)                      # own token

        if self.has_opponent:
            if opp_ctx is None:
                opp_ctx = torch.zeros(B, N, self.opp_proj[0].in_features,
                                      device=h.device)
            h = torch.cat([h, self.opp_proj(opp_ctx)], dim=-1)
        z = self.trunk(h)
        return self.head_cont(z), self.head_disc(z)


# ---------------------------------------------------------------------------
# centralised critic
# ---------------------------------------------------------------------------
class CentralCritic(nn.Module):
    """Q_phi(s~, a) for every agent, with masked agent/device attention."""

    def __init__(self, d_model: int = 64, n_heads: int = 4, n_layers: int = 2,
                 dim_agent: int = 5, dim_dev: int = 5, dim_cont: int = 3,
                 n_mode: int = 3):
        super().__init__()
        self.emb_agent = nn.Linear(dim_agent + dim_cont + n_mode, d_model)
        self.emb_dev = nn.Linear(dim_dev, d_model)
        self.cross = MaskedAttention(d_model, n_heads)
        self.blocks = nn.ModuleList([Block(d_model, n_heads) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = mlp([d_model, d_model, 1], gain=0.01)

    def forward(self, obs, act_cont, act_onehot):
        x = torch.cat([obs["agent_hd"], act_cont, act_onehot], dim=-1)
        x = self.emb_agent(x)                                          # [B,N,D]
        dev = self.emb_dev(obs["dev_hd"])                              # [B,M,D]
        ctx = self.cross(x, dev, bias=obs["bias_ud"], pad=obs["dev_present"])
        h = x + ctx
        for blk in self.blocks:
            h = blk(h, bias=obs["bias_uu"])
        return self.head(self.norm(h)).squeeze(-1)                     # [B,N]


# ---------------------------------------------------------------------------
# implicit opponent model (RNN)
# ---------------------------------------------------------------------------
class OpponentModel(nn.Module):
    """GRU inference of peer policies from observed actions (Sec. IV-C).

    The hidden state ``h_t`` summarises the observed action history of the
    neighbours; it is the opponent context consumed by the actor and it is
    trained with a next-action prediction loss (TBPTT, truncated window).
    """

    def __init__(self, dim_in: int = 6, hidden: int = 32, dim_out: int = 6):
        super().__init__()
        self.hidden = hidden
        self.gru = nn.GRUCell(dim_in, hidden)
        self.head = nn.Linear(hidden, dim_out)

    def step(self, x, h):
        h = self.gru(x, h)
        return h, self.head(h)

    def forward(self, xs, h0):
        """xs: [B, W, dim_in]; returns predictions [B, W, dim_out] and h."""
        outs = []
        h = h0
        for t in range(xs.shape[1]):
            h, pred = self.step(xs[:, t], h)
            outs.append(pred)
        return torch.stack(outs, dim=1), h


__all__ = ["DualHeadActor", "CentralCritic", "CentralValue", "OpponentModel",
           "MaskedAttention", "Block", "mlp"]


class CentralValue(nn.Module):
    """Centralised state-value function V(s~) used by the MAPPO baselines.

    Same masked-attention backbone as :class:`CentralCritic`, but the action is
    not an input: PPO needs ``V(s)`` for the advantage, not ``Q(s, a)``.
    """

    def __init__(self, d_model: int = 64, n_heads: int = 4, n_layers: int = 2,
                 dim_agent: int = 5, dim_dev: int = 5):
        super().__init__()
        self.emb_agent = nn.Linear(dim_agent, d_model)
        self.emb_dev = nn.Linear(dim_dev, d_model)
        self.cross = MaskedAttention(d_model, n_heads)
        self.blocks = nn.ModuleList([Block(d_model, n_heads) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = mlp([d_model, d_model, 1], gain=0.01)

    def forward(self, obs):
        x = self.emb_agent(obs["agent_hd"])                            # [B,N,D]
        dev = self.emb_dev(obs["dev_hd"])                              # [B,M,D]
        ctx = self.cross(x, dev, bias=obs["bias_ud"], pad=obs["dev_present"])
        h = x + ctx
        for blk in self.blocks:
            h = blk(h, bias=obs["bias_uu"])
        return self.head(self.norm(h)).squeeze(-1)                     # [B,N]


__all__ = ["DualHeadActor", "CentralCritic", "CentralValue", "OpponentModel",
           "MaskedAttention", "Block", "mlp"]
