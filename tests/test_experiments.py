"""Unit tests for the experiment harness (CPU only, seconds).

Run with ``pytest -q`` from the repository root.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mapa_exp import baselines
from mapa_exp.config import Config, EnvConfig
from mapa_exp.env import UAVMECEnv
from mapa_exp.mappo import MAPPOAgent
from mapa_exp.metrics import (bootstrap_ci, episodes_to_threshold, paired_bootstrap,
                              relative_deviation, smooth)
from mapa_exp.agent import MAPAAgent


def _tiny_cfg() -> Config:
    cfg = Config.load("quick")
    cfg.train.device = "cpu"
    cfg.train.d_model = 16
    cfg.train.n_heads = 2
    cfg.train.n_layers = 1
    cfg.train.critic_d_model = 16
    cfg.train.critic_n_layers = 1
    cfg.train.rnn_hidden = 8
    cfg.train.batch_size = 8
    cfg.train.warmup_steps = 2
    cfg.train.buffer_episodes = 4
    cfg.train.tbptt = 4
    cfg.env = EnvConfig(n_uav=2, n_device=4, n_device_max=6, n_slot=4)
    cfg.env.task_min, cfg.env.task_max = 2e5, 4e5
    return cfg


# ---------------------------------------------------------------------------
# environment
# ---------------------------------------------------------------------------
def test_env_shapes_and_metrics():
    cfg = _tiny_cfg()
    env = UAVMECEnv(cfg.env, seed=0)
    obs = env.reset()
    assert obs["self"].shape == (cfg.env.n_uav, UAVMECEnv.DIM_SELF)
    _, info = env.step(np.full(2, 0.5), np.tile([1.0, 0.0], (2, 1)),
                       np.array([0, 1]))
    assert 0.0 <= info["qos"] <= 1.0
    assert 0.0 <= info["jfi"] <= 1.0
    assert info["energy"] > 0.0


def test_env_burst_adds_devices():
    cfg = _tiny_cfg()
    env = UAVMECEnv(cfg.env, seed=1)
    env.reset()
    before = int(env._active.sum())
    assert env.activate_devices(2) == 2
    assert int(env._active.sum()) == before + 2


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def test_bootstrap_ci_brackets_the_mean():
    rng = np.random.default_rng(0)
    mat = rng.normal(size=(10, 50)).cumsum(axis=1)
    mean, lo, hi = bootstrap_ci(mat, n_boot=500)
    assert mean.shape == (50,) and lo.shape == (50,) and hi.shape == (50,)
    assert np.all(lo <= mean + 1e-9) and np.all(mean <= hi + 1e-9)


def test_episodes_to_threshold_on_a_synthetic_curve():
    curve = np.concatenate([np.linspace(-30, -12, 100), np.full(100, -12.0)])
    ep = episodes_to_threshold(curve)
    assert 90 <= ep <= 120


def test_paired_bootstrap_detects_a_consistent_difference():
    a = np.array([1.0, 1.1, 0.9, 1.05, 0.95])
    b = np.array([0.0, 0.1, -0.1, 0.05, -0.05])
    out = paired_bootstrap(a, b, n_boot=1000)
    assert out["mean_diff"] == pytest.approx(1.0, abs=1e-6)
    assert out["significant"]


def test_relative_deviation():
    assert relative_deviation(110, 100) == pytest.approx(10.0)
    assert relative_deviation(90, 100) == pytest.approx(-10.0)


def test_smooth_preserves_length():
    x = np.arange(100.0)
    assert smooth(x, 11).shape == x.shape


# ---------------------------------------------------------------------------
# variants
# ---------------------------------------------------------------------------
def test_every_variant_builds_and_has_an_agent_class():
    for name in baselines.VARIANTS:
        cfg = baselines.build_config(name, "quick")
        assert cfg.train.variant == name
        assert baselines.VARIANTS[name].agent in ("mapa", "mappo")


def test_ablation_overrides_are_applied():
    assert baselines.build_config("w_o_zgr", "quick").train.estimator == "stgs"
    assert baselines.build_config("w_o_mask", "quick").train.use_mask is False
    assert baselines.build_config("w_o_pid", "quick").train.use_pid is False
    assert baselines.build_config("maddpg", "quick").train.use_rnn is False


def test_make_agent_returns_the_right_class():
    cfg = _tiny_cfg()
    assert isinstance(baselines.make_agent("mapa_marl", cfg, "cpu", 0), MAPAAgent)
    assert isinstance(baselines.make_agent("mappo_native", cfg, "cpu", 0), MAPPOAgent)


# ---------------------------------------------------------------------------
# MAPPO
# ---------------------------------------------------------------------------
def test_mappo_rollout_shapes_and_gae():
    cfg = _tiny_cfg()
    cfg.train.variant = "mappo_native"
    agent = MAPPOAgent(cfg, device="cpu", seed=0)
    env = UAVMECEnv(cfg.env, seed=0)
    obs = env.reset()
    agent.begin_episode()
    while not env.done:
        action = agent.act(obs, explore=True)
        assert action["speed"].shape == (cfg.env.n_uav,)
        assert action["direction"].shape == (cfg.env.n_uav, 2)
        assert action["mode"].shape == (cfg.env.n_uav,)
        nobs, info = env.step(action["speed"], action["direction"], action["mode"])
        agent.push(obs, action, info, nobs, env.done)
        obs = nobs
    assert agent.end_episode() is None          # not enough steps for a PPO update
    assert len(agent._buffer) == 1


def test_mappo_ppo_update_produces_finite_losses():
    cfg = _tiny_cfg()
    cfg.train.variant = "mappo_native"
    agent = MAPPOAgent(cfg, device="cpu", seed=0)
    agent.ppo["max_steps"] = 4                   # force an update after one episode
    env = UAVMECEnv(cfg.env, seed=0)
    obs = env.reset()
    agent.begin_episode()
    while not env.done:
        action = agent.act(obs, explore=True)
        nobs, info = env.step(action["speed"], action["direction"], action["mode"])
        agent.push(obs, action, info, nobs, env.done)
        obs = nobs
    out = agent.end_episode()
    assert isinstance(out, dict)
    for key in ("pi_loss", "v_loss", "entropy"):
        assert np.isfinite(out[key])


def test_stgs_variant_runs_one_update():
    cfg = _tiny_cfg()
    cfg.train.variant = "mappo_stgs"
    cfg.train.estimator = "stgs"
    agent = MAPPOAgent(cfg, device="cpu", seed=0)
    agent.ppo["max_steps"] = 4
    env = UAVMECEnv(cfg.env, seed=0)
    obs = env.reset()
    agent.begin_episode()
    while not env.done:
        action = agent.act(obs, explore=True)
        nobs, info = env.step(action["speed"], action["direction"], action["mode"])
        agent.push(obs, action, info, nobs, env.done)
        obs = nobs
    out = agent.end_episode()
    assert isinstance(out, dict) and np.isfinite(out["pi_loss"])


# ---------------------------------------------------------------------------
# device utilities (must be safe without a GPU)
# ---------------------------------------------------------------------------
def test_cuda_helpers_degrade_gracefully_on_cpu():
    from mapa_exp.cuda import (amp_context, device_info, maybe_compile,
                               memory_report, resolve_device, timed)

    dev = resolve_device("cpu")
    assert dev.type == "cpu"
    assert device_info(dev)["name"] == "cpu"

    with amp_context(dev, True):          # no-op on CPU, must not raise
        pass

    import torch
    module = torch.nn.Linear(4, 2)
    assert maybe_compile(module, False) is module
    assert memory_report(dev) == {}

    stats = timed(lambda: module(torch.zeros(1, 4)), dev, iters=3, warmup=1)
    assert stats["p50_ms"] >= 0.0
    assert stats["throughput"] > 0.0


# ---------------------------------------------------------------------------
# integration: observation packing, determinism, end-to-end run, public API
# ---------------------------------------------------------------------------
def test_obs_to_tensors_packs_without_changing_values():
    from mapa_exp.agent import obs_to_tensors

    cfg = _tiny_cfg()
    env = UAVMECEnv(cfg.env, seed=3)
    obs = env.reset()
    tensors = obs_to_tensors(obs, cfg, torch.device("cpu"))
    for key in ("self", "peer", "dev", "agent_hd", "dev_hd", "dist_uu",
                "dist_ud", "peer_act", "peer_mask", "dev_mask"):
        reference = torch.as_tensor(np.asarray(obs[key], dtype=np.float32)).unsqueeze(0)
        assert torch.allclose(tensors[key], reference), key


def test_agent_is_deterministic_for_a_fixed_seed():
    cfg = _tiny_cfg()
    a = MAPAAgent(cfg, device="cpu", seed=7)
    b = MAPAAgent(cfg, device="cpu", seed=7)
    obs = UAVMECEnv(cfg.env, seed=0).reset()
    act_a, act_b = a.act(obs, explore=False), b.act(obs, explore=False)
    for key in ("speed", "direction", "mode"):
        assert np.allclose(act_a[key], act_b[key])
    assert act_a["speed"].shape == (cfg.env.n_uav,)
    assert act_a["mode"].dtype == np.int64


def test_training_run_end_to_end():
    import shutil
    from pathlib import Path

    from mapa_exp.train import run

    out_dir = Path("runs/_test_train")          # sandbox-safe location
    cfg = _tiny_cfg()
    cfg.train.episodes = 2
    cfg.train.warmup_steps = 4
    cfg.train.eval_every = 1
    try:
        out = run(cfg, seed=0, out_dir=out_dir, verbose=False)
        assert len(out["history"]) == 2
        assert (out_dir / "history.csv").exists()
        assert (out_dir / "model.pt").exists()
        assert np.isfinite(out["history"][-1]["reward"])
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def test_public_api_surface():
    import mapa_exp

    for name in ("Config", "EnvConfig", "TrainConfig", "UAVMECEnv", "MAPAAgent",
                 "MAPPOAgent", "make_agent", "build_config", "VARIANTS",
                 "obs_to_tensors", "__version__"):
        assert hasattr(mapa_exp, name), name
