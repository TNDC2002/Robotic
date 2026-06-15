"""
Navigate a quadrotor from a fixed spawn to a random goal on a bounded map.

Episode ends on floor crash (low altitude), reaching the goal, or time limit (optional: excessive tilt).

``SAFE_ATTITUDE_DEG`` is the maximum **tilt from vertical**: angle between body +Z and world +Z
(``tilt_deg = acos(uprightness)``), not independent Euler roll/pitch limits.
Exceeding it adds a per-step penalty; staying within earns a per-step bonus.
Set ``NAV_END_ON_UNSAFE_ATTITUDE=1`` in ``.env`` to end the episode on excessive tilt.

**Actions** (``action_mode``):

- ``"motors"`` (default): 4 numbers in [-1, 1] → thrust (N) at each of the four
  rotors (front-left, front-right, back-left, back-right). Set via ``NAV_ACTION_MODE=motors``.
- ``"mixer"``: 4 numbers in [-1, 1] → ``[thrust_delta, roll_mix, pitch_mix, yaw_mix]``
  (not 2D — same action size, different meaning). ``a0=0`` + zero mix = hover (~m·g total).
  Set via ``NAV_ACTION_MODE=mixer``.

Gymnasium wrapper: ``QuadNavGymEnv`` for Stable-Baselines3 / similar trainers.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pybullet as p

from quad_drone_sim import MASS, allocate_motor_thrusts

# Nav spawn is higher than hover demo default to give more altitude margin before floor crash.
DEFAULT_NAV_SPAWN_Z = 1.25
from quad_hover_env import DT, EnvConfig, QuadHoverEnv, WindConfig
from wind_settings import load_wind_settings, sample_episode_wind


def action_to_motor_thrust_n(
    action: float,
    thrust_min_n: float,
    thrust_max_n: float,
) -> float:
    """Linear map action in [-1, 1] to per-motor thrust in [thrust_min_n, thrust_max_n] (N).

    Equal sensitivity across the action range, e.g. for 0–4 N:
      -1 → 0 N, -0.5 → 1 N, 0 → 2 N, 0.5 → 3 N, 1 → 4 N.
    """
    a = max(-1.0, min(1.0, float(action)))
    return thrust_min_n + (thrust_max_n - thrust_min_n) * (a + 1.0) * 0.5


def sync_max_motor_scale(cfg: "NavEnvConfig") -> None:
    """Set physics per-motor cap to ``motor_thrust_max`` (N) for motors and mixer modes."""
    mg = MASS * 9.81
    if cfg.motor_thrust_max <= 0.0:
        raise ValueError(f"motor_thrust_max must be > 0, got {cfg.motor_thrust_max}")
    cfg.max_motor_scale = cfg.motor_thrust_max / mg


@dataclass
class NavMapConfig:
    """Axis-aligned flight region on the ground plane (meters)."""

    x_min: float = -8.0
    x_max: float = 8.0
    y_min: float = -8.0
    y_max: float = 8.0
    cruise_z: float = DEFAULT_NAV_SPAWN_Z
    z_min: float = 0.35
    z_max: float = 1.65


@dataclass
class NavEnvConfig(EnvConfig):
    """Navigation task settings (extends hover physics config)."""

    map: NavMapConfig = field(default_factory=NavMapConfig)
    spawn_xy: tuple[float, float] = (0.0, 0.0)
    spawn_z: float = DEFAULT_NAV_SPAWN_Z
    min_goal_distance: float = 2.5
    goal_radius: float = 0.32
    goal_z_tolerance: float = 0.25
    # Goal altitude is sampled in [spawn_z, spawn_z + goal_z_max_rise] (never below spawn).
    goal_z_max_rise: float = 0.35
    frame_skip: int = 4  # physics substeps per agent step
    max_episode_steps: int = 14_400  # physics steps ≈ 60 s at 240 Hz
    action_mode: Literal["motors", "mixer"] = "motors"
    # Per-motor thrust clip (N): MOTOR_THRUST_MIN_N / MOTOR_THRUST_MAX_N in .env (both action modes).
    max_motor_scale: float = 2.0  # overwritten by sync_max_motor_scale → motor_thrust_max / (m·g)
    # mixer mode only (action in [-1, 1])
    thrust_delta_scale: float = 0.35
    attitude_mix_scale: float = 0.85
    yaw_mix_scale: float = 0.35
    # motors mode: linear map action in [-1, 1] -> thrust in [motor_thrust_min, motor_thrust_max]
    motor_thrust_min: float = 0.0   # action = -1
    motor_thrust_max: float = 8.0   # action = +1  (action = 0 -> midpoint 4 N)
    # Rewards
    w_progress: float = 8.0
    w_goal_dist: float = 0.4
    w_goal_alt: float = 1.0
    w_goal_alt_progress: float = 4.0
    w_alive: float = 0.02
    w_attitude: float = 1.2
    w_ang_vel: float = 0.06
    w_lin_vel: float = 0.04
    w_upright: float = 0.15
    w_action_rate: float = 0.02
    w_unsafe_attitude: float = 5.0  # per substep when tilt >= flip_angle_rad
    w_safe_attitude: float = 1.0  # per substep when tilt < flip_angle_rad
    success_bonus: float = 120.0
    crash_penalty: float = 80.0
    time_limit_penalty: float = 15.0
    end_on_unsafe_attitude: bool = False  # terminate when tilt >= SAFE_ATTITUDE_DEG
    penalty_unsafe_attitude_end: float = 50.0  # terminal penalty if end_on_unsafe_attitude
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
        self._prev_goal_alt_err = 0.0
        self._prev_action = np.zeros(4, dtype=np.float64)
        self._goal_marker: int | None = None
        self._episode_rng = np.random.default_rng()
        self._fixed_wind: WindConfig | None = None

    def connect(self) -> None:
        sync_max_motor_scale(self.nav)
        super().connect()

    def _wind_in_observation(self) -> bool:
        settings = self.nav.wind_settings
        if settings is None or not settings.enabled:
            return False
        return bool(settings.include_in_obs)

    @property
    def observation_size(self) -> int:
        return 18 + (3 if self._wind_in_observation() else 0)

    def _sample_episode_wind(self) -> WindConfig:
        if self._fixed_wind is not None:
            return self._fixed_wind
        settings = self.nav.wind_settings or load_wind_settings()
        return sample_episode_wind(self._episode_rng, settings)

    def _resample_episode_wind(self) -> None:
        self.cfg.wind = self._sample_episode_wind()
        self._wind_turbulence = (0.0, 0.0, 0.0)
        self._wind_rng = random.Random(self.cfg.wind.seed)
        self._sync_wind_visualize_flag()

    def _goal_z_range(self) -> tuple[float, float]:
        spawn_z = self.nav.spawn_z
        gz_lo = spawn_z
        gz_hi = min(self.nav.map.z_max, spawn_z + self.nav.goal_z_max_rise)
        if gz_hi < gz_lo:
            gz_hi = gz_lo
        return gz_lo, gz_hi

    def _sample_goal(self) -> tuple[float, float, float]:
        m = self.nav.map
        sx, sy = self.nav.spawn_xy
        gz_lo, gz_hi = self._goal_z_range()
        for _ in range(64):
            gx = self._episode_rng.uniform(m.x_min, m.x_max)
            gy = self._episode_rng.uniform(m.y_min, m.y_max)
            gz = self._episode_rng.uniform(gz_lo, gz_hi)
            if math.hypot(gx - sx, gy - sy) >= self.nav.min_goal_distance:
                return (gx, gy, gz)
        # Fallback: point along +X if sampling fails.
        return (sx + self.nav.min_goal_distance, sy, gz_lo)

    def _goal_distance(self, pos: tuple[float, float, float]) -> float:
        return math.sqrt(
            (pos[0] - self._goal_pos[0]) ** 2
            + (pos[1] - self._goal_pos[1]) ** 2
            + (pos[2] - self._goal_pos[2]) ** 2
        )

    def _goal_alt_error(self, pos: tuple[float, float, float]) -> float:
        return abs(pos[2] - self._goal_pos[2])

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
        elif self.nav.wind_settings is not None and self.nav.wind_settings.seed is not None:
            self._episode_rng = np.random.default_rng(self.nav.wind_settings.seed)
        self._spawn_pos = (
            self.nav.spawn_xy[0],
            self.nav.spawn_xy[1],
            self.nav.spawn_z,
        )
        self._goal_pos = self._sample_goal()
        spawn = pos if pos is not None else self._spawn_pos
        state = super().reset(pos=spawn, orn=orn)
        self._clear_force_visualization()
        self._update_goal_marker()
        self._start_goal_dist = self._goal_distance(state.pos)
        self._prev_goal_dist = self._start_goal_dist
        self._prev_goal_alt_err = self._goal_alt_error(state.pos)
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
        if self._wind_in_observation() and self.cfg.wind.enabled:
            wx, wy, wz = self._effective_wind_velocity(self._step_count * DT)
            obs.extend([wx, wy, wz])
        return np.array(obs, dtype=np.float32)

    @staticmethod
    def tilt_rad_from_uprightness(uprightness: float) -> float:
        """Angle (rad) between body +Z and world +Z; 0 = level, pi = inverted."""
        return math.acos(max(-1.0, min(1.0, uprightness)))

    @staticmethod
    def _uprightness_from_orn(orn: tuple[float, float, float, float]) -> float:
        from quad_hover_env import body_z_in_world

        tip = body_z_in_world(orn)
        return tip[2]

    def _clamp_motor_thrusts(self, thrusts: list[float]) -> list[float]:
        lo = self.nav.motor_thrust_min
        hi = self.nav.motor_thrust_max
        return [min(hi, max(lo, float(t))) for t in thrusts]

    def _decode_action_mixer(self, action: np.ndarray) -> list[float]:
        a = np.clip(action, -1.0, 1.0)
        g = 9.81
        t_max = self.nav.motor_thrust_max
        thrust_sum = MASS * g * (1.0 + self.nav.thrust_delta_scale * a[0])
        thrust_sum = max(0.0, min(4.0 * t_max, thrust_sum))
        mix_r = self.nav.attitude_mix_scale * a[1]
        mix_p = self.nav.attitude_mix_scale * a[2]
        thrusts = allocate_motor_thrusts(thrust_sum, mix_r, mix_p, self._signs_xy, t_max)
        if abs(a[3]) > 1e-6:
            yaw_delta = self.nav.yaw_mix_scale * a[3] * (MASS * g * 0.08)
            thrusts[0] = min(t_max, thrusts[0] + yaw_delta)
            thrusts[1] = min(t_max, thrusts[1] + yaw_delta)
            thrusts[2] = max(0.0, thrusts[2] - yaw_delta)
            thrusts[3] = max(0.0, thrusts[3] - yaw_delta)
        return self._clamp_motor_thrusts(thrusts)

    def _decode_action_motors(self, action: np.ndarray) -> list[float]:
        """Map 4D policy output to per-rotor thrust (N).

        Linear, equal sensitivity over [-1, 1]:
          thrust_N = min + (max - min) * (action + 1) / 2
          e.g. min=0, max=4: -1→0 N, -0.5→1 N, 0→2 N, 0.5→3 N, 1→4 N

        Motor order matches ``signs_xy`` / corner layout:
          0 = +X +Y, 1 = +X -Y, 2 = -X +Y, 3 = -X -Y (body frame).
        """
        a = np.clip(action, -1.0, 1.0)
        t_min = self.nav.motor_thrust_min
        t_max = self.nav.motor_thrust_max
        return self._clamp_motor_thrusts([
            float(action_to_motor_thrust_n(a[i], t_min, t_max))
            for i in range(4)
        ])

    def _decode_action(self, action: np.ndarray | list[float]) -> list[float]:
        arr = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        if self.nav.action_mode == "motors":
            return self._decode_action_motors(arr)
        return self._decode_action_mixer(arr)

    def _apply_forces(
        self,
        action: list[float] | tuple[float, ...] | None,
        *,
        update_viz: bool = True,
    ) -> None:
        assert self._drone is not None
        pos, orn = p.getBasePositionAndOrientation(self._drone)
        lin_vel, _ = p.getBaseVelocity(self._drone)

        if self.cfg.hover_balance_thrust:
            mg4 = MASS * 9.81 / 4.0
            thrusts = [mg4, mg4, mg4, mg4]
            self._apply_motor_thrusts(pos, orn, thrusts)
            self._apply_wind(pos, orn, lin_vel)
            if update_viz:
                self._update_force_visualization(pos, pos, orn, lin_vel)
            return

        if action is None:
            raise ValueError("QuadNavEnv requires an RL action each step")
        thrusts = self._decode_action(action)  # type: ignore[arg-type]
        self._apply_motor_thrusts(pos, orn, thrusts)
        self._apply_wind(pos, orn, lin_vel)
        if update_viz:
            self._update_force_visualization(pos, pos, orn, lin_vel)

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

        alt_err = self._goal_alt_error(pos)
        alt_progress = self._prev_goal_alt_err - alt_err
        self._prev_goal_alt_err = alt_err

        tilt_rad = self.tilt_rad_from_uprightness(uprightness)

        parts = {
            "progress": self.nav.w_progress * progress,
            "goal_dist": -self.nav.w_goal_dist * dist,
            "goal_alt": -self.nav.w_goal_alt * alt_err,
            "goal_alt_progress": self.nav.w_goal_alt_progress * alt_progress,
            "alive": self.nav.w_alive,
            "attitude": -self.nav.w_attitude * tilt_rad,
            "ang_vel": -self.nav.w_ang_vel * math.sqrt(sum(w * w for w in ang_vel)),
            "lin_vel": -self.nav.w_lin_vel * math.sqrt(sum(v * v for v in lin_vel)),
            "upright": self.nav.w_upright * max(0.0, uprightness),
            "action_rate": -self.nav.w_action_rate * float(np.sum((action - self._prev_action) ** 2)),
        }
        if self._attitude_unsafe(uprightness):
            parts["unsafe_attitude"] = -self.nav.w_unsafe_attitude
        else:
            parts["safe_attitude"] = self.nav.w_safe_attitude
        terminal = 0.0
        if done:
            if terminal_reason == "success":
                terminal = self.nav.success_bonus
            elif terminal_reason == "crash":
                terminal = -self.nav.crash_penalty
            elif terminal_reason == "unsafe_attitude":
                terminal = -self.nav.penalty_unsafe_attitude_end
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

        for sub_i in range(max(1, self.nav.frame_skip)):
            last_sub = sub_i == max(1, self.nav.frame_skip) - 1
            self._apply_forces(action_arr, update_viz=last_sub)
            p.stepSimulation()
            self._step_count += 1

            pos, orn = p.getBasePositionAndOrientation(self._drone)
            lin_vel, ang_vel = p.getBaseVelocity(self._drone)
            roll, pitch, yaw = p.getEulerFromQuaternion(orn)
            uprightness = self._uprightness_from_orn(orn)
            done, terminal_reason = self._check_termination_simple(pos, uprightness=uprightness)
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

        if self.cfg.step_sleep_s > 0.0:
            import time

            time.sleep(self.cfg.step_sleep_s)

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
            "motor_thrusts": (
                [MASS * 9.81 / 4.0] * 4
                if self.cfg.hover_balance_thrust
                else self._decode_action(action_arr)
            ),
            "unsafe_attitude": self._attitude_unsafe(uprightness),
            "tilt_deg": math.degrees(self.tilt_rad_from_uprightness(uprightness)),
            "uprightness": uprightness,
        }
        if self._last_wind_info is not None:
            info["wind"] = self._last_wind_info
        elif self.cfg.wind.enabled:
            info["wind"] = {
                "velocity": self.cfg.wind.velocity,
                "drag_coeff": self.cfg.wind.drag_coeff,
                "turbulence_std": self.cfg.wind.turbulence_std,
                "force_noise_std": self.cfg.wind.force_noise_std,
            }
        return obs, total_reward, done, info

    def _attitude_unsafe(self, uprightness: float) -> bool:
        return self.tilt_rad_from_uprightness(uprightness) >= self.cfg.flip_angle_rad

    def _check_termination_simple(
        self,
        pos: tuple[float, float, float],
        *,
        uprightness: float,
    ) -> tuple[bool, str | None]:
        if pos[2] <= self.cfg.crash_z:
            return True, "crash"
        if (
            self.nav.end_on_unsafe_attitude
            and self.tilt_rad_from_uprightness(uprightness) >= self.cfg.flip_angle_rad
        ):
            return True, "unsafe_attitude"
        dist_xy = math.hypot(pos[0] - self._goal_pos[0], pos[1] - self._goal_pos[1])
        dist_z = abs(pos[2] - self._goal_pos[2])
        if dist_xy <= self.nav.goal_radius and dist_z <= self.nav.goal_z_tolerance:
            return True, "success"
        if not self.cfg.unlimited_episode and self._step_count >= self.nav.max_episode_steps:
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
    motor_thrust_min: float | None = None,
    motor_thrust_max: float | None = None,
) -> QuadNavEnv:
    from env_config import build_nav_env_config

    cfg = build_nav_env_config(
        gui=gui,
        step_sleep_s=step_sleep_s,
        motor_thrust_min=motor_thrust_min,
        motor_thrust_max=motor_thrust_max,
    )
    env = QuadNavEnv(cfg)
    if wind is not None:
        env._fixed_wind = wind
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
            self._completed_outcomes: list[dict[str, Any]] = []
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
            if terminated or truncated:
                self._completed_outcomes.append(
                    {"terminal_reason": info.get("terminal_reason")}
                )
            return obs, reward, terminated, truncated, info

        def drain_completed_outcomes(self) -> list[dict[str, Any]]:
            outcomes = self._completed_outcomes
            self._completed_outcomes = []
            return outcomes

        def close(self):
            self._env.close()

except ImportError:
    QuadNavGymEnv = None  # type: ignore[misc, assignment]
