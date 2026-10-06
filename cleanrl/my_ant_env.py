"""AntSpin-v0: teach the MuJoCo Ant to spin in place about the vertical axis.

Import this module before calling gym.make("AntSpin-v0") so the register() call runs.

Diagnostics (off by default): set ANTSPIN_DIAG_LOG=/path/file.csv to append one row per episode end
with the reason it ended, torso height and uprightness.
"""
import os

import numpy as np
from gymnasium.envs.mujoco.ant_v4 import AntEnv
from gymnasium.envs.registration import register

MAX_EPISODE_STEPS = 1000
# Upright = cos(angle between the torso's up-axis and world up): 1 level, 0 on its side, -1 upside down.
# 0.5 means tilted by at most 60 degrees.
MIN_UPRIGHT = 0.5


def _yaw_from_quat(quat):
    """World-frame heading (rotation about z) from a MuJoCo (w, x, y, z) quaternion."""
    w, x, y, z = quat
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _upright_from_quat(quat):
    _, x, y, _ = quat
    return 1.0 - 2.0 * (x * x + y * y)


class AntSpinEnv(AntEnv):
    def __init__(self, spin_weight=1.0, drift_weight=0.5, **kwargs):
        # Default Ant ends episodes when torso z leaves [0.2, 1.0]. That ceiling cut off upright spinners that
        # hopped slightly, while an Ant on its back (z ~0.3) never terminated and learned to spin there.
        # So: raise the ceiling, and add an uprightness check in is_healthy below.
        kwargs.setdefault("healthy_z_range", (0.2, 1.5))
        super().__init__(**kwargs)
        self._spin_weight = spin_weight
        self._drift_weight = drift_weight
        self._prev_yaw = 0.0
        self._cumulative_yaw = 0.0  # for the TRUE metric
        self._t = 0
        self._diag_log = os.environ.get("ANTSPIN_DIAG_LOG")

    @property
    def is_healthy(self):
        # Used by the parent for both termination and healthy_reward.
        return super().is_healthy and _upright_from_quat(self.data.qpos[3:7]) >= MIN_UPRIGHT

    def reset_model(self):
        obs = super().reset_model()
        self._prev_yaw = _yaw_from_quat(self.data.qpos[3:7])
        self._cumulative_yaw = 0.0  # zero the scoreboard each episode
        self._t = 0
        return obs

    def step(self, action):
        # Parent runs the physics, termination check, and builds the standard info dict.
        obs, _, terminated, truncated, info = super().step(action)

        # World-frame yaw rate from the change in heading (wrapped to [-pi, pi]).
        # qvel[5] is body-frame angular velocity, which only equals yaw rate when the torso is level.
        yaw = _yaw_from_quat(self.data.qpos[3:7])
        delta_yaw = (yaw - self._prev_yaw + np.pi) % (2 * np.pi) - np.pi
        self._prev_yaw = yaw
        yaw_rate = delta_yaw / self.dt

        # TRUE METRIC: signed full rotations completed while upright this episode (+ = counter-clockwise).
        if _upright_from_quat(self.data.qpos[3:7]) >= MIN_UPRIGHT:
            self._cumulative_yaw += delta_yaw
        rotations = self._cumulative_yaw / (2 * np.pi)

        # Penalize horizontal torso speed, not distance from start: speed is physically bounded,
        # so the penalty can't grow until it outweighs healthy_reward and make falling over optimal.
        drift_speed = np.hypot(info["x_velocity"], info["y_velocity"])
        ctrl_cost = self.control_cost(action)
        spin_reward = self._spin_weight * yaw_rate
        drift_cost = self._drift_weight * drift_speed

        # Keep healthy_reward: without it every step is net-negative and the agent learns to fall over early.
        reward = spin_reward + self.healthy_reward - drift_cost - ctrl_cost

        info.update(
            {
                "yaw_rate": yaw_rate,
                "drift_speed": drift_speed,
                "reward_spin": spin_reward,
                "reward_drift": -drift_cost,
                "true_metric_rotations": rotations,
            }
        )
        self._t += 1
        if self._diag_log:
            self._log_episode_end(terminated, rotations)
        return obs, reward, terminated, truncated, info

    def _log_episode_end(self, terminated, rotations):
        # The TimeLimit wrapper sits outside this env, so detect truncation from our own step count.
        if not terminated and self._t != MAX_EPISODE_STEPS:
            return
        z = self.data.qpos[2]
        min_z, max_z = self._healthy_z_range
        if not terminated:
            cause = "time_limit"
        elif not np.isfinite(self.state_vector()).all():
            cause = "nonfinite"
        elif z < min_z:
            cause = "too_low"
        elif z > max_z:
            cause = "too_high"
        else:
            cause = "tipped"
        upright = _upright_from_quat(self.data.qpos[3:7])
        with open(self._diag_log, "a") as f:
            f.write(f"{self._t},{cause},{z:.3f},{upright:.3f},{rotations:.3f}\n")


register(
    id="AntSpin-v0",
    entry_point=f"{__name__}:AntSpinEnv",
    max_episode_steps=MAX_EPISODE_STEPS,
)

# The training scripts call gym.make(env_id) with no kwargs, so spin_weight is selected via the id:
# --env-id AntSpin1-v0 / AntSpin2-v0 / AntSpin3-v0 / AntSpin5-v0
for w in (1.0, 2.0, 3.0, 5.0):
    register(
        id=f"AntSpin{w:g}-v0",
        entry_point=f"{__name__}:AntSpinEnv",
        max_episode_steps=MAX_EPISODE_STEPS,
        kwargs={"spin_weight": w},
    )
