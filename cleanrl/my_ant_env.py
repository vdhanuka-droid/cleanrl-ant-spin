"""AntSpin-v0: teach the MuJoCo Ant to spin in place about the vertical axis.

Import this module before calling gym.make("AntSpin-v0") so the register() call runs.
"""
import numpy as np
from gymnasium.envs.mujoco.ant_v4 import AntEnv
from gymnasium.envs.registration import register


def _yaw_from_quat(quat):
    """World-frame heading (rotation about z) from a MuJoCo (w, x, y, z) quaternion."""
    w, x, y, z = quat
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class AntSpinEnv(AntEnv):
    def __init__(self, spin_weight=1.0, drift_weight=0.5, **kwargs):
        super().__init__(**kwargs)
        self._spin_weight = spin_weight
        self._drift_weight = drift_weight
        self._prev_yaw = 0.0
        self._cumulative_yaw = 0.0  # for the TRUE metric

    def reset_model(self):
        obs = super().reset_model()
        self._prev_yaw = _yaw_from_quat(self.data.qpos[3:7])
        self._cumulative_yaw = 0.0  # zero the scoreboard each episode
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

        # TRUE METRIC: signed full rotations completed this episode (+ = counter-clockwise).
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
        return obs, reward, terminated, truncated, info


register(
    id="AntSpin-v0",
    entry_point=f"{__name__}:AntSpinEnv",
    max_episode_steps=1000,
)

# The training scripts call gym.make(env_id) with no kwargs, so spin_weight is selected via the id:
# --env-id AntSpin1-v0 / AntSpin2-v0 / AntSpin3-v0 / AntSpin5-v0
for w in (1.0, 2.0, 3.0, 5.0):
    register(
        id=f"AntSpin{w:g}-v0",
        entry_point=f"{__name__}:AntSpinEnv",
        max_episode_steps=1000,
        kwargs={"spin_weight": w},
    )
