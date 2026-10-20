# API reference

Short tour of the public surface. Everything is importable from the package root
unless stated otherwise.

```python
import mapa_exp
print(mapa_exp.__version__)
```

## Configuration

| symbol | module | notes |
|---|---|---|
| `Config` | `config.py` | root object: `cfg.env`, `cfg.train`; `Config.load("quick"\|"paper")` |
| `EnvConfig` | `config.py` | physics and economics: area, altitude, bandwidth, CPU, energy, thresholds |
| `TrainConfig` | `config.py` | network width, learning rates, PID gains, mask shape, meta options |

`Config.apply_overrides({"env.n_uav": 6, "train.use_mask": False})` mutates by
dotted path — the same mechanism the CLI uses (`--set env.n_uav=6`).

## Simulator

```python
from mapa_exp import UAVMECEnv
env = UAVMECEnv(cfg.env, seed=0)
obs = env.reset()
obs, info = env.step(speed, direction, mode)      # (N,), (N,2), (N,)
```

| member | meaning |
|---|---|
| `obs` | dict of arrays: `self`, `peer`, `dev`, `agent_hd`, `dev_hd`, `dist_uu`, `dist_ud`, `peer_act`, `peer_mask`, `dev_mask` |
| `info` | `cost`, `qos`, `jfi`, `delay`, `energy`, `starvation`, `collisions`, `safety`, mode mix |
| `heuristic_action(policy)` | `"local"`, `"edge"`, `"cloud"`, `*, "-goto"` — reference policies |
| `activate_devices(k)` | task-burst scenario |
| `history()` | trajectories and per-slot modes for plotting |

## Agents

```python
from mapa_exp import MAPAAgent, MAPPOAgent, make_agent, build_config

cfg = build_config("mapa_marl", "quick")     # or "maddpg", "w_o_mask", ...
agent = make_agent("mapa_marl", cfg, device="cuda", seed=0)
```

Both agent classes share one interface, so a training loop is agent-agnostic:

`begin_episode()` → `act(obs, explore=True)` → `push(...)` → `update()` →
`end_episode()` → `reset_hidden()` / `advance_hidden(peer_act)`.

`MAPAAgent.update()` returns per-update diagnostics (critic/actor loss, the two
head gradient norms, opponent-model loss); `MAPPOAgent` updates on-policy at the
end of an episode and returns the PPO losses from `end_episode()`.

## Building blocks

| symbol | module | what it is |
|---|---|---|
| `DualHeadActor` | `networks.py` | attention encoder + continuous and categorical heads |
| `CentralCritic` / `CentralValue` | `networks.py` | masked-attention `Q(s,a)` and `V(s)` |
| `OpponentModel` | `networks.py` | GRU over observed peer actions |
| `MaskedAttention` / `Block` / `mlp` | `networks.py` | the transformer pieces |
| `softplus_mask` / `rbf_mask` / `log_mask_bias` | `mask.py` | spatial regularisers |
| `zgr_surrogate` / `stgs_one_hot` / `anneal_temperature` | `estimators.py` | discrete gradient estimators |
| `PIDLagrangian` | `lagrangian.py` | constrained-RL multiplier |
| `EpisodeBuffer` | `replay.py` | episode storage + TBPTT window sampling |
| `bootstrap_ci` / `episodes_to_threshold` / `paired_bootstrap` | `metrics.py` | statistics |
| `resolve_device` / `timed` / `amp_context` / `maybe_compile` | `cuda.py` | device and timing helpers |

## Variants

`VARIANTS` maps a name to the agent class and the overrides it applies:
`mapa_marl`, `w_o_zgr`, `w_o_mask`, `w_o_pid`, `maddpg`, `maac`, `mrltl`,
`mappo_native`, `mappo_clipped`, `mappo_stgs`.

## Command line

```bash
python -m mapa_exp.train --profile quick --episodes 100 --seed 0 \
    --set train.use_mask=false
python scripts/profile_model.py --profile paper
python scripts/benchmark.py --batch-sizes 1 8 32
```
