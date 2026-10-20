"""Simplified UAV-assisted MEC environment.

Implements the three-tier model of the paper (Sec. III): IoT devices -> UAV
swarm (aerial edge / relay) -> cloud server.

Included physics
----------------
* Probabilistic air-to-ground path loss with LoS/NLoS mixture (Eq. 1).
* Shannon uplink rate (Eq. 2) and the resulting upload delay.
* Velocity dependent propulsion power ``P_hov + kappa v^3`` (Eq. 3) and the
  per-UAV energy breakdown ``E = E_com + E_fly + E_tran`` (Eq. 4).
* Smooth exponential QoS (Eq. 5) and Jain's Fairness Index (Eq. 6).
* Normalised system cost ``Omega`` (Eq. 7) and the soft safety penalty used by
  the demo to reproduce the collision statistics of Table II.

Deliberate simplifications (documented in ``docs/method.md``)
-------------------------------------------------------------
* No queueing: every device generates exactly one task per slot.  The paper
  mentions queue lengths as an observation but does not define the queue
  dynamics, so we keep the state Markov and drop the queue.
* Devices associate to the nearest UAV ("single-user-single-relay"), and the
  UAV's discrete action selects one offloading mode for *all* devices it serves
  (local compute / UAV edge compute / relay to cloud).  This is the atomic
  hybrid-action semantics of Eq. (12).
* Collision and area constraints (Eq. 8c/8d) are handled by projection plus a
  soft penalty instead of a hard constrained layer.

Action space (per UAV, Sec. IV-A, Eq. 12)
-----------------------------------------
``speed``     in [0, 1]        -> v_n = speed * Vmax           (continuous)
``direction`` in R^2 (unit)    -> yaw phi_n = atan2(dy, dx)     (continuous)
``mode``      in {0, 1, 2}     -> {LOCAL, EDGE, CLOUD}          (discrete)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .config import EnvConfig

# discrete offloading decisions, matching k_n in {-1, 0, +1}
LOCAL, EDGE, CLOUD = 0, 1, 2
MODE_NAMES = ("local", "edge", "cloud")


def _db2lin(value: float) -> float:
    return 10.0 ** (value / 10.0)


class UAVMECEnv:
    """Vectorised multi-UAV MEC environment (single-agent view per UAV)."""

    #: feature dimensionalities consumed by the networks
    DIM_SELF = 6
    DIM_PEER = 5
    DIM_DEV = 6
    DIM_AGENT = 5
    DIM_DEV_HIDDEN = 5
    DIM_PEER_ACT = 6

    def __init__(self, cfg: EnvConfig, seed: Optional[int] = None,
                 n_device: Optional[int] = None, task_scale: float = 1.0):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.n = cfg.n_uav
        self.m_max = cfg.n_device_max
        self.n_device = int(n_device if n_device is not None else cfg.n_device)
        if self.n_device > self.m_max:
            raise ValueError("n_device must be <= n_device_max")
        self.task_scale = float(task_scale)
        self.n_slot = cfg.n_slot

        self.beta0 = _db2lin(cfg.beta0_db)
        self.sigma2 = _db2lin(cfg.noise_dbm) / 1000.0
        self.mu_nlos = _db2lin(-cfg.mu_nlos_db)

        self._active = np.zeros(self.m_max, dtype=bool)
        self._peer_idx = np.array([[j for j in range(self.n) if j != i]
                                   for i in range(self.n)], dtype=np.int64)
        self.reset()

    # ==================================================================
    # reset / activation
    # ==================================================================
    def reset(self) -> Dict[str, np.ndarray]:
        cfg, rng = self.cfg, self.rng
        self._active[: self.n_device] = True
        self._active[self.n_device:] = False

        # UAVs start on a coarse grid over the area
        grid = int(np.ceil(np.sqrt(self.n)))
        idx = np.arange(self.n)
        centres = np.stack([((idx % grid) + 0.5) / grid,
                            ((idx // grid) + 0.5) / grid], axis=1)
        self.pos_uav = centres * cfg.area + rng.normal(0, 0.05 * cfg.area, (self.n, 2))
        self.pos_uav = np.clip(self.pos_uav, 0.0, cfg.area)

        self.pos_dev = rng.uniform(0.0, cfg.area, (self.m_max, 2))
        self.waypoint = rng.uniform(0.0, cfg.area, (self.m_max, 2))
        # heterogeneous device CPUs (0.5x - 1.5x nominal) make the LOCAL mode
        # genuinely competitive for small tasks on fast devices
        self.cpu_norm = rng.uniform(0.5, 1.5, self.m_max)
        self.cpu_dev = cfg.cpu_local * self.cpu_norm

        self.prev_speed = np.zeros(self.n, dtype=np.float64)
        self.energy_spent = np.zeros(self.n, dtype=np.float64)
        self.t = 0
        self.served = np.zeros((self.n, self.m_max), dtype=bool)
        self.owner = np.zeros(self.m_max, dtype=np.int64)
        self.task = np.zeros(self.m_max, dtype=np.float64)
        self.qos = np.zeros(self.m_max, dtype=np.float64)
        self.rate = np.zeros((self.n, self.m_max), dtype=np.float64)
        self._peer_act = np.zeros((self.n, self.DIM_PEER_ACT), dtype=np.float32)
        self._last_info: Dict[str, float] = {}
        self._history = {"uav": [self.pos_uav.copy()],
                         "dev": [self.pos_dev.copy()],
                         "mode": []}
        self._sample_tasks()
        self._update_association()
        return self._obs()

    def activate_devices(self, k: int) -> int:
        """Activate ``k`` dormant devices (used by the task-burst scenario)."""
        dormant = np.flatnonzero(~self._active)
        k = int(min(k, dormant.size))
        if k > 0:
            self._active[dormant[:k]] = True
            self.pos_dev[dormant[:k]] = self.rng.uniform(0.0, self.cfg.area, (k, 2))
            self.waypoint[dormant[:k]] = self.rng.uniform(0.0, self.cfg.area, (k, 2))
            self.cpu_norm[dormant[:k]] = self.rng.uniform(0.5, 1.5, k)
            self.cpu_dev[dormant[:k]] = self.cfg.cpu_local * self.cpu_norm[dormant[:k]]
        self.n_device = int(self._active.sum())
        return k

    # ==================================================================
    # physics helpers
    # ==================================================================
    def _dist_uav_dev(self) -> np.ndarray:
        """3D distance [N, M] between UAVs and devices."""
        d2 = self.pos_dev[None, :, :] - self.pos_uav[:, None, :]
        return np.sqrt((d2 ** 2).sum(-1) + self.cfg.height ** 2)

    def _dist_uav_uav(self) -> np.ndarray:
        d = self.pos_uav[:, None, :] - self.pos_uav[None, :, :]
        return np.sqrt((d ** 2).sum(-1))

    def _los_probability(self, dist3d: np.ndarray) -> np.ndarray:
        """Eq. (1): probabilistic LoS link between UAV and device."""
        cfg = self.cfg
        d2 = np.sqrt(np.clip(dist3d ** 2 - cfg.height ** 2, 0.0, None))
        elev = np.degrees(np.arctan2(cfg.height, np.maximum(d2, 1e-6)))
        return 1.0 / (1.0 + cfg.a_los * np.exp(-cfg.b_los * (elev - cfg.a_los)))

    def _channel_gain(self, dist3d: np.ndarray) -> np.ndarray:
        """Expected channel power gain g_{m,n} (Sec. III-B)."""
        cfg = self.cfg
        p_los = self._los_probability(dist3d)
        return (p_los + (1.0 - p_los) * self.mu_nlos) * self.beta0 * dist3d ** (-cfg.alpha)

    def _rate(self, dist3d: np.ndarray) -> np.ndarray:
        """Eq. (2): Shannon uplink rate [bit/s]."""
        snr = self.cfg.tx_power * self._channel_gain(dist3d) / self.sigma2
        return self.cfg.bandwidth * np.log2(1.0 + snr)

    def _sample_tasks(self) -> None:
        lo, hi = self.cfg.task_min, self.cfg.task_max
        self.task = self.rng.uniform(lo, hi, self.m_max) * self.task_scale
        self.task[~self._active] = 0.0

    def _update_association(self) -> None:
        dist = self._dist_uav_dev()                       # [N, M]
        self.owner = np.argmin(dist, axis=0)              # nearest UAV
        self.served = np.zeros((self.n, self.m_max), dtype=bool)
        self.served[self.owner, np.arange(self.m_max)] = True
        self.served[:, ~self._active] = False
        self.rate = self._rate(dist)                      # [N, M]

    # ==================================================================
    # step
    # ==================================================================
    def step(self, speed: np.ndarray, direction: np.ndarray,
             mode: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
        cfg = self.cfg
        speed = np.clip(np.asarray(speed, dtype=np.float64), 0.0, 1.0)
        direction = np.asarray(direction, dtype=np.float64)
        direction = direction / (np.linalg.norm(direction, axis=-1, keepdims=True) + 1e-8)
        mode = np.asarray(mode, dtype=np.int64).reshape(-1)
        self.mode = mode

        # ---- UAV kinematics (projection onto the area, Eq. 8c) --------
        v = speed * cfg.v_max
        self.pos_uav = np.clip(self.pos_uav + direction * (v * cfg.dt)[:, None],
                               0.0, cfg.area)
        self.prev_speed = speed

        # ---- device mobility: random waypoint -------------------------
        delta = self.waypoint - self.pos_dev
        norm = np.linalg.norm(delta, axis=-1, keepdims=True) + 1e-8
        step_len = cfg.device_speed * cfg.dt
        arrived = (norm.squeeze(-1) <= step_len) & self._active
        self.pos_dev = np.where(arrived[:, None], self.waypoint,
                                self.pos_dev + delta / norm * step_len)
        if arrived.any():
            self.waypoint[arrived] = self.rng.uniform(0.0, cfg.area,
                                                      (int(arrived.sum()), 2))

        # ---- task generation & association ----------------------------
        self._sample_tasks()
        self._update_association()

        # ---- delay / energy (Eq. 2-5) ---------------------------------
        bits = self.task                                          # [M]
        bits_per_uav = (self.served * bits[None, :]).sum(1)       # [N]

        t_edge = bits_per_uav * cfg.cycles_per_bit / cfg.cpu_uav
        t_cloud = (cfg.backhaul_delay
                   + bits_per_uav / cfg.backhaul_rate
                   + bits_per_uav * cfg.cycles_per_bit / cfg.cpu_cloud)
        t_local_task = bits * cfg.cycles_per_bit / np.maximum(self.cpu_dev, 1e6)

        rate_owner = self.rate[self.owner, np.arange(self.m_max)]  # [M]
        t_upload = np.where(self._active, bits / np.maximum(rate_owner, 1.0), 0.0)

        delay = np.where(
            mode[self.owner] == LOCAL, t_local_task,
            np.where(mode[self.owner] == EDGE,
                     t_upload + t_edge[self.owner],
                     t_upload + t_cloud[self.owner]),
        )
        delay = np.where(self._active, delay, 0.0)

        # ---- QoS (Eq. 5) and Jain's index (Eq. 6) ---------------------
        lateness = np.maximum(0.0, delay - cfg.t_max)
        qos = np.exp(-cfg.theta * lateness)
        self.qos = qos.copy()                                # per-device QoS (plots)
        qos_act = qos[self._active]
        mean_delay = float(delay[self._active].mean())
        jfi = float((qos_act.sum() ** 2) /
                    (qos_act.size * (qos_act ** 2).sum() + 1e-12))
        starvation = float((qos_act < 0.5).mean())

        # ---- energy (Eq. 3-4) -----------------------------------------
        e_fly = (cfg.p_hover + cfg.kappa_drag * v ** 3) * cfg.dt
        e_com = np.where(mode == EDGE, cfg.p_uav_compute * t_edge, 0.0)
        e_tran = np.where(mode == CLOUD, cfg.p_tx * bits_per_uav / cfg.backhaul_rate, 0.0)
        energy = e_fly + e_com + e_tran
        self.energy_spent += energy

        # ---- safety (Eq. 8d) ------------------------------------------
        dist_uu = self._dist_uav_uav()
        iu = np.triu_indices(self.n, k=1)
        separations = dist_uu[iu]
        violation = np.clip(cfg.d_min - separations, 0.0, None) / cfg.d_min
        collisions = float((separations < cfg.d_min).sum())
        safety = float(violation.sum())

        # ---- normalised system cost (Eq. 7) ---------------------------
        omega = (cfg.w_delay * mean_delay / cfg.t_max
                 + cfg.w_energy * float(energy.sum()) / (self.n * cfg.energy_ref))

        # ---- bookkeeping ---------------------------------------------
        self._history["uav"].append(self.pos_uav.copy())
        self._history["dev"].append(self.pos_dev.copy())
        self._history["mode"].append(mode.copy())
        self._peer_act = self._peer_action_summary(mode, speed, direction)
        self.t += 1

        info = {
            "cost": float(omega),
            "jfi": jfi,
            "qos": float(qos_act.mean()),
            "starvation": starvation,
            "delay": mean_delay,
            "energy": float(energy.sum()),
            "safety": safety,
            "collisions": collisions,
            "bits": float(bits_per_uav.sum()),
            "mode_edge": float((mode == EDGE).mean()),
            "mode_cloud": float((mode == CLOUD).mean()),
            "mode_local": float((mode == LOCAL).mean()),
        }
        self._last_info = info
        return self._obs(), info

    def _peer_action_summary(self, mode: np.ndarray, speed: np.ndarray,
                             direction: np.ndarray) -> np.ndarray:
        """Compact description of the joint action, used by the RNN opponent model."""
        onehot = np.eye(3)[mode]                              # [N, 3]
        vec = np.concatenate([speed[:, None], direction, onehot], axis=1)  # [N, 6]
        out = np.zeros((self.n, self.DIM_PEER_ACT), dtype=np.float32)
        for i in range(self.n):
            peers = vec[self._peer_idx[i]]
            out[i] = peers.mean(0) if peers.size else 0.0
        return out

    # ==================================================================
    # observation
    # ==================================================================
    def _obs(self) -> Dict[str, np.ndarray]:
        cfg = self.cfg
        dist = self._dist_uav_dev()                          # [N, M]
        # NOTE: mask parameters live in TrainConfig; the environment exposes the
        # raw distances and the agent builds the masks with its own (ablated)
        # settings, so "w/o Sparse-Mask" only changes the network, not the world.
        d2 = self.pos_dev[None, :, :] - self.pos_uav[:, None, :]        # [N, M, 2]
        active = self._active.astype(np.float32)

        task_frac = np.broadcast_to(self.task[None, :] / cfg.task_max, dist.shape)
        cpu_frac = np.broadcast_to((self.cpu_norm[None, :] - 0.5), dist.shape)
        dev = np.stack([
            d2[..., 0] / cfg.area,
            d2[..., 1] / cfg.area,
            task_frac,
            dist / cfg.area,
            np.broadcast_to(active[None, :], dist.shape),
            cpu_frac,
        ], axis=-1).astype(np.float32)                                  # [N, M, 6]

        energy_frac = 1.0 - self.energy_spent / cfg.battery
        served_bits = (self.served * self.task[None, :]).sum(1)
        self_feat = np.stack([
            self.pos_uav[:, 0] / cfg.area,
            self.pos_uav[:, 1] / cfg.area,
            self.prev_speed,
            energy_frac,
            np.full(self.n, self.t / self.n_slot),
            served_bits / (max(self.n_device, 1) * cfg.task_max),
        ], axis=-1).astype(np.float32)                                  # [N, 6]

        # peer view: fixed slot order, self removed
        duu = self._dist_uav_uav()
        peer = np.zeros((self.n, self.n - 1, self.DIM_PEER), dtype=np.float32)
        for i in range(self.n):
            idx = self._peer_idx[i]
            peer[i, :, 0] = (self.pos_uav[idx, 0] - self.pos_uav[i, 0]) / cfg.area
            peer[i, :, 1] = (self.pos_uav[idx, 1] - self.pos_uav[i, 1]) / cfg.area
            peer[i, :, 2] = self.prev_speed[idx]
            peer[i, :, 3] = energy_frac[idx]
            peer[i, :, 4] = duu[i, idx] / cfg.area
        peer_mask = np.ones((self.n, self.n - 1), dtype=np.float32)

        agent_hd = np.stack([
            self.pos_uav[:, 0] / cfg.area,
            self.pos_uav[:, 1] / cfg.area,
            self.prev_speed,
            energy_frac,
            served_bits / (max(self.n_device, 1) * cfg.task_max),
        ], axis=-1).astype(np.float32)                                  # [N, 5]

        dev_hd = np.stack([
            self.pos_dev[:, 0] / cfg.area,
            self.pos_dev[:, 1] / cfg.area,
            self.task / cfg.task_max,
            active,
            self.cpu_norm - 0.5,
        ], axis=-1).astype(np.float32)                                  # [M, 5]

        return {
            "self": self_feat,
            "peer": peer,
            "peer_mask": peer_mask,
            "dev": dev,
            "dev_mask": np.broadcast_to(active[None, :], (self.n, self.m_max)).copy(),
            "agent_hd": agent_hd,
            "dev_hd": dev_hd,
            "dist_uu": duu.astype(np.float32),            # for the mask
            "dist_ud": dist.astype(np.float32),           # for the mask
            "peer_act": self._peer_act.copy(),
        }

    # ==================================================================
    # utilities
    # ==================================================================
    @property
    def done(self) -> bool:
        return self.t >= self.n_slot

    def history(self) -> Dict[str, np.ndarray]:
        return {k: np.asarray(v) for k, v in self._history.items()}

    def heuristic_action(self, policy: str = "edge") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simple baselines used for calibration and sanity checks."""
        cfg = self.cfg
        if policy.endswith("-goto"):
            # move towards the nearest unserved device cluster centroid
            served = self.served.any(0)
            targets = self.pos_dev[served] if served.any() else self.pos_dev[self._active]
            direction = np.zeros((self.n, 2))
            for i in range(self.n):
                diff = targets - self.pos_uav[i]
                d = np.linalg.norm(diff, axis=-1) + 1e-6
                w = 1.0 / d
                direction[i] = (diff * w[:, None]).sum(0) / w.sum()
            speed = np.full(self.n, 0.6)
        else:
            direction = np.tile(np.array([1.0, 0.0]), (self.n, 1))
            speed = np.zeros(self.n)
        base = policy.replace("-goto", "")
        mode = np.full(self.n, {"local": LOCAL, "edge": EDGE, "cloud": CLOUD}[base],
                       dtype=np.int64)
        return speed, direction, mode
