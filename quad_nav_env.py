"""
Navigate a quadrotor from a fixed spawn to a random goal on a bounded map.

Episode ends on crash (low altitude or flip) or reaching the goal. Reward encourages
progress toward the goal while staying level and smooth.

**Actions** (``action_mode``):

- ``"motors"`` (default): 4 numbers in [-1, 1] → thrust (N) at each of the four
  rotors (front-left, front-right, back-left, back-right). This matches four engines.
- ``"mixer"``: 4 numbers → total thrust + roll/pitch/yaw mixing (easier to learn).

Gymnasium wrapper: ``QuadNavGymEnv`` for Stable-Baselines3 / similar trainers.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pybullet as p

from quad_drone_sim import HOVER_Z, MASS, allocate_motor_thrusts, scale_vec
from quad_hover_env import DT, EnvConfig, QuadHoverEnv, WindConfig


@dataclass
class NavMapConfig:
    """Axis-aligned flight region on the ground plane (meters)."""

    x_min: float = -8.0
    x_max: float = 8.0
    y_min: float = -8.0
    y_max: float = 8.0
    cruise_z: float = HOVER_Z
    z_min: float = 0.35
    z_max: float = 1.15


@dataclass
class NavWindRandomizationConfig:
    """Per-episode random wind (enabled by default for navigation training)."""

    enabled: bool = True
    speed_range: tuple[float, float] = (0.5, 2.5)  # m/s, random horizontal direction
    drag_range: tuple[float, float] = (0.35, 0.75)
    quad_drag_range: tuple[float, float] = (0.03, 0.08)
    # Multiplier on base turbulence (0.4, 0.4, 0.3); 20 ≈ demo --wind-turbulence 20
    turbulence_scale_range: tuple[float, float] = (4.0, 22.0)
    force_noise_range: tuple[float, float] = (0.2, 4.0)
    corner_noise_range: tuple[float, float] = (0.06, 0.35)
    torque_noise_range: tuple[float, float] = (0.03, 0.15)
    gust_amplitude_range: tuple[float, float] = (0.0, 1.0)
    base_turbulence_std: tuple[float, float, float] = (0.4, 0.4, 0.3)


@dataclass
class NavEnvConfig(EnvConfig):
    """Navigation task settings (extends hover physics config)."""

    map: NavMapConfig = field(default_factory=NavMapConfig)
    spawn_xy: tuple[float, float] = (0.0, 0.0)
    spawn_z: float = HOVER_Z
    min_goal_distance: float = 2.5
    goal_radius: float = 0.32
    goal_z_tolerance: float = 0.25
    frame_skip: int = 4  # physics substeps per agent step
    max_episode_steps: int = 14_400  # physics steps ≈ 60 s at 240 Hz
    action_mode: Literal["motors", "mixer"] = "motors"
    wind_randomization: NavWindRandomizationConfig = field(default_factory=NavWindRandomizationConfig)
    # mixer mode only (action in [-1, 1])
    thrust_delta_scale: float = 0.35
    attitude_mix_scale: float = 0.85
    yaw_mix_scale: float = 0.35
    # motors mode: thrust_i = hover_per_motor * (1 + motor_span * a_i), clipped to [0, max_motor]
    motor_hover_span: float = 0.85
    # Rewards
    w_progress: float = 8.0
    w_goal_dist: float = 0.4
    w_alive: float = 0.02
    w_attitude: float = 1.2
    w_ang_vel: float = 0.06
    w_lin_vel: float = 0.04
    w_upright: float = 0.15
    w_action_rate: float = 0.02
    success_bonus: float = 120.0
    crash_penalty: float = 80.0
    time_limit_penalty: float = 15.0
    show_goal_marker: bool = True


class QuadNavEnv(QuadHoverEnv):
    """Spawn → random goal navigation with stability-shaped reward."""

    def __init__(self, config: NavEnvConfig | None = None) -> None:
        super().__init__(config or NavEnvConfig())
        self.nav: NavEnvConfig = self.cfg  # type: ignore[assignment]
        self._goal_pos = (0.0, 0.0, self.nav.map.cruise_z)
        self._spawn_pos = (self.nav.spawn_xy[0], self.nav.spawn_xy[1], self.nav.spawn_z)
        self._start_goal_dist = 1.0
        self._prev_goal_dist = 1.0
        self._prev_action = np.zeros(4, dtype=np.float64)
        self._goal_marker: int | None = None
        self._episode_rng = np.random.default_rng()

    @property
    def observation_size(self) -> int:
        extra = 3 if self.cfg.wind.enabled and self.cfg.wind.include_in_obs else 0
        return 18 + extra

    def _sample_episode_wind(self) -> WindConfig:
        """Random wind each episode (strength similar to aggressive demo CLI)."""
        wr = self.nav.wind_randomization
        if not wr.enabled:
            return WindConfig(
                enabled=self.nav.wind.enabled,
                velocity=self.nav.wind.velocity,
                drag_coeff=self.nav.wind.drag_coeff,
                quad_drag_coeff=self.nav.wind.quad_drag_coeff,
                turbulence_std=self.nav.wind.turbulence_std,
                force_noise_std=self.nav.wind.force_noise_std,
                corner_force_noise_std=self.nav.wind.corner_force_noise_std,
                torque_noise_std=self.nav.wind.torque_noise_std,
                gust_amplitude=self.nav.wind.gust_amplitude,
                include_in_obs=self.nav.wind.include_in_obs,
            )

        speed = float(self._episode_rng.uniform(*wr.speed_range))
        angle = float(self._episode_rng.uniform(0.0, 2.0 * math.pi))
        turb_scale = float(self._episode_rng.uniform(*wr.turbulence_scale_range))
        turb = tuple(turb_scale * t for t in wr.base_turbulence_std)
        return WindConfig(
            enabled=True,
            velocity=(speed * math.cos(angle), speed * math.sin(angle), 0.0),
            drag_coeff=float(self._episode_rng.uniform(*wr.drag_range)),
            quad_drag_coeff=float(self._episode_rng.uniform(*wr.quad_drag_range)),
            turbulence_std=turb,
            force_noise_std=float(self._episode_rng.uniform(*wr.force_noise_range)),
            corner_force_noise_std=float(self._episode_rng.uniform(*wr.corner_noise_range)),
            torque_noise_std=float(self._episode_rng.uniform(*wr.torque_noise_range)),
            gust_amplitude=float(self._episode_rng.uniform(*wr.gust_amplitude_range)),
            include_in_obs=True,
            seed=int(self._episode_rng.integers(0, 2**31 - 1)),
        )

    def _sample_goal(self) -> tuple[float, float, float]:
        m = self.nav.map
        sx, sy = self._spawn_pos[0], self._spawn_pos[1]
        for _ in range(64):
            gx = self._episode_rng.uniform(m.x_min, m.x_max)
            gy = self._episode_rng.uniform(m.y_min, m.y_max)
            gz = self._episode_rng.uniform(
                max(m.z_min, m.cruise_z - self.nav.goal_z_tolerance),
                min(m.z_max, m.cruise_z + self.nav.goal_z_tolerance),
            )
            if math.hypot(gx - sx, gy - sy) >= self.nav.min_goal_distance:
                return (gx, gy, gz)
        # Fallback: point along +X if sampling fails.
        return (sx + self.nav.min_goal_distance, sy, m.cruise_z)

    def _goal_distance(self, pos: tuple[float, float, float]) -> float:
        return math.sqrt(
            (pos[0] - self._goal_pos[0]) ** 2
            + (pos[1] - self._goal_pos[1]) ** 2
            + (pos[2] - self._goal_pos[2]) ** 2
        )

    def _update_goal_marker(self) -> None:
        if not self.nav.show_goal_marker or self._drone is None:
            return
        if self._goal_marker is not None:
            p.removeBody(self._goal_marker)
        radius = self.nav.goal_radius
        vis = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=radius,
            rgbaColor=(0.95, 0.25, 0.15, 0.45),
        )
        self._goal_marker = p.createMultiBody(
            0,
            -1,
            vis,
            basePosition=self._goal_pos,
        )

    def reset(
        self,
        *,
        pos: tuple[float, float, float] | None = None,
        orn: tuple[float, float, float, float] | None = None,
        seed: int | None = None,
    ) -> np.ndarray:
        if seed is not None:
            self._episode_rng = np.random.default_rng(seed)
        self._goal_pos = self._sample_goal()
        self._spawn_pos = (
            self.nav.spawn_xy[0],
            self.nav.spawn_xy[1],
            self.nav.spawn_z,
        )
        self.cfg.wind = self._sample_episode_wind()
        self._wind_turbulence = (0.0, 0.0, 0.0)
        self._wind_rng = random.Random(self.cfg.wind.seed)
        spawn = pos if pos is not None else self._spawn_pos
        state = super().reset(pos=spawn, orn=orn)
        self._update_goal_marker()
        self._start_goal_dist = self._goal_distance(state.pos)
        self._prev_goal_dist = self._start_goal_dist
        self._prev_action = np.zeros(4, dtype=np.float64)
        return self.observation_vector_from_physics()

    def observation_vector_from_physics(self) -> np.ndarray:
        assert self._drone is not None
        pos, orn = p.getBasePositionAndOrientation(self._drone)
        lin_vel, ang_vel = p.getBaseVelocity(self._drone)
        roll, pitch, yaw = p.getEulerFromQuaternion(orn)

        rel = (
            self._goal_pos[0] - pos[0],
            self._goal_pos[1] - pos[1],
            self._goal_pos[2] - pos[2],
        )
        dist = self._goal_distance(pos)
        start = max(self._start_goal_dist, 1e-3)

        obs: list[float] = [
            rel[0],
            rel[1],
            rel[2],
            dist / start,
            lin_vel[0],
            lin_vel[1],
            lin_vel[2],
            roll,
            pitch,
            math.sin(yaw),
            math.cos(yaw),
            ang_vel[0],
            ang_vel[1],
            ang_vel[2],
            max(-1.0, min(1.0, self._uprightness_from_orn(orn))),
            self._spawn_pos[0],
            self._spawn_pos[1],
            self._goal_pos[0],
        ]
        if self.cfg.wind.enabled and self.cfg.wind.include_in_obs:
            wx, wy, wz = self._wind_velocity_world(self._step_count * DT)
            obs.extend([wx, wy, wz])
        return np.array(obs, dtype=np.float32)

    @staticmethod
    def _uprightness_from_orn(orn: tuple[float, float, float, float]) -> float:
        from quad_hover_env import body_z_in_world

        tip = body_z_in_world(orn)
        return tip[2]

    def _decode_action_mixer(self, action: np.ndarray) -> list[float]:
        a = np.clip(action, -1.0, 1.0)
        g = 9.81
        thrust_sum = MASS * g * (1.0 + self.nav.thrust_delta_scale * a[0])
        thrust_sum = max(0.0, min(4.0 * self._max_motor, thrust_sum))
        mix_r = self.nav.attitude_mix_scale * a[1]
        mix_p = self.nav.attitude_mix_scale * a[2]
        thrusts = allocate_motor_thrusts(thrust_sum, mix_r, mix_p, self._signs_xy, self._max_motor)
        if abs(a[3]) > 1e-6:
            yaw_delta = self.nav.yaw_mix_scale * a[3] * (MASS * g * 0.08)
            thrusts[0] = min(self._max_motor, thrusts[0] + yaw_delta)
            thrusts[1] = min(self._max_motor, thrusts[1] + yaw_delta)
            thrusts[2] = max(0.0, thrusts[2] - yaw_delta)
            thrusts[3] = max(0.0, thrusts[3] - yaw_delta)
        return thrusts

    def _decode_action_motors(self, action: np.ndarray) -> list[float]:
        """Map 4D policy output to per-rotor thrust (N).

        Motor order matches ``signs_xy`` / corner layout:
          0 = +X +Y, 1 = +X -Y, 2 = -X +Y, 3 = -X -Y (body frame).
        """
        a = np.clip(action, -1.0, 1.0)
        hover = MASS * 9.81 / 4.0
        span = self.nav.motor_hover_span
        return [
            float(min(self._max_motor, max(0.0, hover * (1.0 + span * a[i]))))
            for i in range(4)
        ]

    def _decode_action(self, action: np.ndarray | list[float]) -> list[float]:
        arr = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        if self.nav.action_mode == "motors":
            return self._decode_action_motors(arr)
        return self._decode_action_mixer(arr)

    def _apply_forces(self, action: list[float] | tuple[float, ...] | None) -> None:
        assert self._drone is not None
        if action is None:
            raise ValueError("QuadNavEnv requires an RL action each step")
        thrusts = self._decode_action(action)  # type: ignore[arg-type]
        pos, orn = p.getBasePositionAndOrientation(self._drone)
        lin_vel, _ = p.getBaseVelocity(self._drone)
        from quad_hover_env import body_z_in_world

        thrust_dir = body_z_in_world(orn)
        from quad_hover_env import world_from_body

        for corner_body, thrust in zip(self._corners_body, thrusts):
            world_point = world_from_body(pos, orn, corner_body)
            force = scale_vec(thrust_dir, thrust)
            p.applyExternalForce(self._drone, -1, force, world_point, p.WORLD_FRAME)
        self._apply_wind(pos, orn, lin_vel)

    def _compute_nav_reward(
        self,
        *,
        pos: tuple[float, float, float],
        roll: float,
        pitch: float,
        lin_vel: tuple[float, float, float],
        ang_vel: tuple[float, float, float],
        uprightness: float,
        action: np.ndarray,
        done: bool,
        terminal_reason: str | None,
    ) -> tuple[float, dict[str, float]]:
        dist = self._goal_distance(pos)
        progress = self._prev_goal_dist - dist
        self._prev_goal_dist = dist

        parts = {
            "progress": self.nav.w_progress * progress,
            "goal_dist": -self.nav.w_goal_dist * dist,
            "alive": self.nav.w_alive,
            "attitude": -self.nav.w_attitude * (abs(roll) + abs(pitch)),
            "ang_vel": -self.nav.w_ang_vel * math.sqrt(sum(w * w for w in ang_vel)),
            "lin_vel": -self.nav.w_lin_vel * math.sqrt(sum(v * v for v in lin_vel)),
            "upright": self.nav.w_upright * max(0.0, uprightness),
            "action_rate": -self.nav.w_action_rate * float(np.sum((action - self._prev_action) ** 2)),
        }
        terminal = 0.0
        if done:
            if terminal_reason == "success":
                terminal = self.nav.success_bonus
            elif terminal_reason == "crash":
                terminal = -self.nav.crash_penalty
            elif terminal_reason == "time_limit":
                terminal = -self.nav.time_limit_penalty
        parts["terminal"] = terminal
        total = sum(parts.values())
        return total, parts

    def step(
        self,
        action: np.ndarray | list[float],
    ) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        assert self._drone is not None
        action_arr = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        terminal_reason = None
        total_reward = 0.0
        last_parts: dict[str, float] = {}

        for _ in range(max(1, self.nav.frame_skip)):
            self._apply_forces(action_arr)
            p.stepSimulation()
            self._step_count += 1
            if self.cfg.step_sleep_s > 0.0:
                import time

                time.sleep(self.cfg.step_sleep_s)

            pos, orn = p.getBasePositionAndOrientation(self._drone)
            lin_vel, ang_vel = p.getBaseVelocity(self._drone)
            roll, pitch, yaw = p.getEulerFromQuaternion(orn)
            uprightness = self._uprightness_from_orn(orn)
            done, terminal_reason = self._check_termination_simple(pos, roll, pitch)
            step_reward, last_parts = self._compute_nav_reward(
                pos=pos,
                roll=roll,
                pitch=pitch,
                lin_vel=lin_vel,
                ang_vel=ang_vel,
                uprightness=uprightness,
                action=action_arr,
                done=done,
                terminal_reason=terminal_reason,
            )
            total_reward += step_reward
            if done:
                break

        self._prev_action = action_arr.copy()
        obs = self.observation_vector_from_physics()
        info: dict[str, Any] = {
            "terminal_reason": terminal_reason,
            "goal": self._goal_pos,
            "spawn": self._spawn_pos,
            "distance_to_goal": self._goal_distance(pos),
            "reward_parts": last_parts,
            "step": self._step_count,
            "action_mode": self.nav.action_mode,
            "motor_thrusts": self._decode_action(action_arr),
        }
        if self.cfg.wind.enabled:
            info["wind"] = {
                "velocity": self.cfg.wind.velocity,
                "drag_coeff": self.cfg.wind.drag_coeff,
                "turbulence_std": self.cfg.wind.turbulence_std,
                "force_noise_std": self.cfg.wind.force_noise_std,
            }
        return obs, total_reward, done, info

    def _check_termination_simple(
        self,
        pos: tuple[float, float, float],
        roll: float,
        pitch: float,
    ) -> tuple[bool, str | None]:
        if pos[2] <= self.cfg.crash_z:
            return True, "crash"
        if abs(roll) >= self.cfg.flip_angle_rad or abs(pitch) >= self.cfg.flip_angle_rad:
            return True, "crash"
        dist_xy = math.hypot(pos[0] - self._goal_pos[0], pos[1] - self._goal_pos[1])
        dist_z = abs(pos[2] - self._goal_pos[2])
        if dist_xy <= self.nav.goal_radius and dist_z <= self.nav.goal_z_tolerance:
            return True, "success"
        if self._step_count >= self.nav.max_episode_steps:
            return True, "time_limit"
        return False, None

    def close(self) -> None:
        if self._goal_marker is not None:
            try:
                p.removeBody(self._goal_marker)
            except Exception:
                pass
            self._goal_marker = None
        super().close()


def make_nav_env(
    *,
    gui: bool = False,
    wind: WindConfig | None = None,
    seed: int | None = None,
    step_sleep_s: float = 0.0,
) -> QuadNavEnv:
    cfg = NavEnvConfig(gui=gui, step_sleep_s=step_sleep_s)
    if wind is not None:
        cfg.wind = wind
    env = QuadNavEnv(cfg)
    if seed is not None:
        env._episode_rng = np.random.default_rng(seed)
    return env


try:
    import gymnasium as gym
    from gymnasium import spaces

    class QuadNavGymEnv(gym.Env):
        """Gymnasium API wrapper around :class:`QuadNavEnv`."""

        metadata = {"render_modes": []}

        def __init__(self, config: NavEnvConfig | None = None, render_mode: str | None = None):
            super().__init__()
            self._env = QuadNavEnv(config)
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self._env.observation_size,),
                dtype=np.float32,
            )
            self.render_mode = render_mode

        def reset(self, *, seed: int | None = None, options: dict | None = None):
            super().reset(seed=seed)
            if seed is not None:
                self._env._episode_rng = np.random.default_rng(seed)
            obs = self._env.reset(seed=seed)
            info = {
                "goal": self._env._goal_pos,
                "spawn": self._env._spawn_pos,
                "action_mode": self._env.nav.action_mode,
            }
            if self._env.cfg.wind.enabled:
                info["wind"] = self._env.cfg.wind.velocity
            return obs, info

        def step(self, action):
            obs, reward, terminated, info = self._env.step(action)
            truncated = info.get("terminal_reason") == "time_limit"
            return obs, reward, terminated, truncated, info

        def close(self):
            self._env.close()

except ImportError:
    QuadNavGymEnv = None  # type: ignore[misc, assignment]
