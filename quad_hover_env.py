"""
Gym-style quad hover environment for reinforcement learning (PyBullet).

Task (v1): stay near a fixed hover pose and avoid flip/crash.
No RL library dependency — ``step`` returns state, reward, done, and info.

Example:
    env = QuadHoverEnv(gui=True)
    state = env.reset()
    state, reward, done, info = env.step()
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

import pybullet as p

from quad_drone_sim import (
    DT,
    GUI_STEP_SLEEP_S,
    HALF_EXTENTS,
    HOVER_Z,
    MASS,
    REALTIME_STEP_SLEEP_S,
    allocate_motor_thrusts,
    body_z_in_world,
    create_drone_visual_shape,
    load_ground_plane,
    motor_corners_local,
    scale_vec,
    world_from_body,
)


SIM_FREQ_HZ = 1.0 / DT


def episode_steps_for_seconds(seconds: float) -> int:
    """Convert wall-clock episode length to simulation frames at 240 Hz."""
    return max(1, int(round(seconds * SIM_FREQ_HZ)))


@dataclass
class WindConfig:
    """Wind field + aerodynamic drag (PyBullet has no built-in wind).

    Effective air velocity = mean ``velocity`` + sinusoidal gust + 3D turbulence.
    Drag uses linear + optional quadratic terms; forces are applied at each motor
    corner with per-corner noise and optional torque noise so gusts can tilt the drone.
    """

    enabled: bool = False
    velocity: tuple[float, float, float] = (0.3, 0.0, 0.0)  # mean wind (m/s), world frame
    drag_coeff: float = 0.3  # linear drag scale (N·s/m per axis)
    quad_drag_coeff: float = 0.04  # extra |v|·v drag per axis (N·s²/m²); stronger in gusts
    gust_amplitude: float = 0.0  # sinusoidal gust magnitude on wind.x (m/s)
    gust_freq_hz: float = 0.15
    vertical_gust_coupling: float = 0.12  # fraction of gust added to vertical wind (was hardcoded 0.35)
    # Smoothed 3D turbulence (Ornstein–Uhlenbeck); affects X, Y, and Z wind components.
    turbulence_std: tuple[float, float, float] = (0.4, 0.4, 0.08)
    turbulence_tau_s: float = 0.3  # correlation time; lower = jerkier gusts
    force_noise_std: float = 0.12  # extra random force (N) per axis on total drag
    force_noise_z_scale: float = 0.25  # multiplier on vertical force noise (lower = less altitude spam)
    corner_force_noise_std: float = 0.08  # random force (N) per axis at each motor corner
    corner_force_noise_z_scale: float = 0.25
    torque_noise_std: float = 0.04  # random torque (N·m) per axis
    seed: int | None = None  # None = nondeterministic turbulence
    include_in_obs: bool = True  # append effective wind_xyz to observation when enabled
    # GUI debug arrows (PyBullet addUserDebugLine); enable via EnvConfig.show_wind_visualization.
    visualize: bool = False
    viz_grid_half_extent: float = 1.0  # draw a small drag field grid around the drone
    viz_grid_spacing: float = 1.0
    viz_update_stride: int = 8  # refresh arrows every N physics steps (GUI only)
    viz_show_applied_drag: bool = False  # include noisy applied drag (flickers; off by default)
    viz_show_weight: bool = True  # purple arrow = weight (N), for thrust comparison


@dataclass
class EnvConfig:
    """Tunable physics, targets, termination, and reward weights."""

    target_xy: tuple[float, float] = (0.0, 0.0)
    target_z: float = HOVER_Z
    # One step() == one p.stepSimulation() at DT (240 Hz). 2400 steps ≈ 10 s.
    max_episode_steps: int = 2400
    flip_angle_rad: float = math.radians(70.0)
    crash_z: float = 0.12
    max_xy: float = 2.5
    gui: bool = False
    # GUI playback: no max_episode_steps timeout (episode ends on task failure/success only).
    unlimited_episode: bool = False
    # Wall-clock pause after each stepSimulation (0 = as fast as possible).
    step_sleep_s: float = 0.0
    show_wind_visualization: bool = True  # drag-force arrows in GUI when wind is enabled
    show_thrust_visualization: bool = True  # green thrust arrows at each motor in GUI
    # Shared scale for force arrows (N): thrust, weight, and drag when force_viz_wind_mode=drag_n.
    force_viz_length_per_n: float = 0.12
    force_viz_min_length: float = 0.04
    force_viz_max_length: float = 1.2
    force_viz_update_stride: int = 1
    # Wind arrows: drag (N) by default; wind_ms shows air velocity (m/s) with longer arrows.
    force_viz_wind_mode: Literal["drag_n", "wind_ms"] = "drag_n"
    force_viz_length_per_ms: float = 3.0
    wind: WindConfig = field(default_factory=WindConfig)
    wind_settings: "WindSettings | None" = None
    # PD hover controller (used when action is None)
    kp_z: float = 18.0
    kd_z: float = 4.5
    kp_rp: float = 0.45
    kd_rp: float = 0.08
    max_motor_scale: float = 0.65  # fraction of mg per motor cap
    # Fixed mg/4 per motor (no PD / no RL action) — wind-tunnel visualization mode.
    hover_balance_thrust: bool = False
    # Reward weights
    w_alive: float = 0.05
    w_pos_xy: float = 1.0
    w_pos_z: float = 1.5
    w_attitude: float = 2.0
    w_lin_vel: float = 0.15
    w_ang_vel: float = 0.08
    w_upright_bonus: float = 0.25
    penalty_flip: float = 50.0
    penalty_crash: float = 30.0
    penalty_oob: float = 40.0

    @property
    def episode_duration_s(self) -> float:
        return self.max_episode_steps / SIM_FREQ_HZ


@dataclass
class DroneState:
    """What the agent observes after each physics step."""

    step: int
    pos: tuple[float, float, float]
    lin_vel: tuple[float, float, float]
    roll: float
    pitch: float
    yaw: float
    ang_vel: tuple[float, float, float]
    uprightness: float  # body +Z dotted with world +Z; 1 = level, 0 = on side
    target: tuple[float, float, float]
    err_xy: float
    err_z: float
    err_roll: float
    err_pitch: float
    wind_world: tuple[float, float, float] | None = None

    def as_vector(self) -> list[float]:
        """Flat observation for future RL policies."""
        obs = [
            self.pos[0],
            self.pos[1],
            self.pos[2],
            self.lin_vel[0],
            self.lin_vel[1],
            self.lin_vel[2],
            self.roll,
            self.pitch,
            self.yaw,
            self.ang_vel[0],
            self.ang_vel[1],
            self.ang_vel[2],
            self.uprightness,
            self.err_xy,
            self.err_z,
            self.err_roll,
            self.err_pitch,
        ]
        if self.wind_world is not None:
            obs.extend(self.wind_world)
        return obs

    def labels(self) -> list[str]:
        labels = [
            "x (m)",
            "y (m)",
            "z (m)",
            "vx (m/s)",
            "vy (m/s)",
            "vz (m/s)",
            "roll (rad)",
            "pitch (rad)",
            "yaw (rad)",
            "wx (rad/s)",
            "wy (rad/s)",
            "wz (rad/s)",
            "uprightness",
            "xy error (m)",
            "z error (m)",
            "roll error (rad)",
            "pitch error (rad)",
        ]
        if self.wind_world is not None:
            labels.extend(["wind_x (m/s)", "wind_y (m/s)", "wind_z (m/s)"])
        return labels


@dataclass
class RewardBreakdown:
    total: float
    alive: float
    position: float
    attitude: float
    velocity: float
    upright_bonus: float
    terminal_penalty: float
    terminal_reason: str | None = None

    def explain(self) -> str:
        parts = [
            f"total={self.total:+.3f}",
            f"alive={self.alive:+.3f}",
            f"position={self.position:+.3f}",
            f"attitude={self.attitude:+.3f}",
            f"velocity={self.velocity:+.3f}",
            f"upright_bonus={self.upright_bonus:+.3f}",
        ]
        if self.terminal_penalty:
            parts.append(f"terminal={self.terminal_penalty:+.3f} ({self.terminal_reason})")
        return ", ".join(parts)


class QuadHoverEnv:
    """PyBullet env: stable hover at a fixed pose; penalize drift, tilt, and flip."""

    def __init__(self, config: EnvConfig | None = None) -> None:
        self.cfg = config or EnvConfig()
        self._client: int | None = None
        self._drone: int | None = None
        self._corners_body: list[tuple[float, float, float]] = []
        self._signs_xy = [(+1.0, +1.0), (+1.0, -1.0), (-1.0, +1.0), (-1.0, -1.0)]
        self._max_motor = 0.0
        self._step_count = 0
        self._roll_trim = 0.0
        self._pitch_trim = 0.0
        self._last_wind_info: dict[str, Any] | None = None
        self._wind_turbulence = (0.0, 0.0, 0.0)
        self._wind_rng = random.Random(self.cfg.wind.seed)
        self._force_viz_line_ids: list[int] = []
        self._force_viz_text_id: int = -1
        self._last_thrusts: list[float] = [0.0, 0.0, 0.0, 0.0]
        self._last_thrust_dir: tuple[float, float, float] = (0.0, 0.0, 1.0)
        self._episode_rng = np.random.default_rng()

    def _resample_episode_wind(self) -> None:
        from wind_settings import load_wind_settings, sample_episode_wind

        settings = self.cfg.wind_settings or load_wind_settings()
        self.cfg.wind_settings = settings
        if settings.seed is not None:
            self._episode_rng = np.random.default_rng(settings.seed)
        self.cfg.wind = sample_episode_wind(self._episode_rng, settings)
        self._wind_turbulence = (0.0, 0.0, 0.0)
        self._wind_rng = random.Random(self.cfg.wind.seed)
        self._sync_wind_visualize_flag()

    def _sync_wind_visualize_flag(self) -> None:
        self.cfg.wind.visualize = bool(
            self.cfg.gui and self.cfg.wind.enabled and self.cfg.show_wind_visualization
        )

    @staticmethod
    def _force_magnitude_color(force_n: float, ref_n: float = 10.0) -> list[float]:
        """Map force magnitude (N) to RGB for debug lines."""
        t = max(0.0, min(1.0, force_n / max(1e-6, ref_n)))
        return [t, 0.45 + 0.4 * (1.0 - t), 1.0 - t]

    @staticmethod
    def _force_viz_scale(cfg: EnvConfig) -> tuple[float, float, float]:
        return (cfg.force_viz_length_per_n, cfg.force_viz_min_length, cfg.force_viz_max_length)

    @staticmethod
    def _wind_viz_scale(cfg: EnvConfig) -> tuple[float, float, float]:
        return (cfg.force_viz_length_per_ms, cfg.force_viz_min_length, cfg.force_viz_max_length)

    @staticmethod
    def _vec_add(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
        return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

    @staticmethod
    def _vec_scale(v: tuple[float, float, float], s: float) -> tuple[float, float, float]:
        return (v[0] * s, v[1] * s, v[2] * s)

    @staticmethod
    def _vec_unit(v: tuple[float, float, float]) -> tuple[float, float, float]:
        n = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
        if n <= 1e-9:
            return (1.0, 0.0, 0.0)
        return (v[0] / n, v[1] / n, v[2] / n)

    @staticmethod
    def _vec_cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
        return (
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        )

    def _clear_debug_lines(self, line_ids: list[int]) -> None:
        if self._client is None or not p.isConnected(self._client):
            line_ids.clear()
            return
        for uid in line_ids:
            try:
                p.removeUserDebugItem(uid)
            except Exception:
                pass
        line_ids.clear()

    def _clear_force_visualization(self) -> None:
        if self._client is not None and p.isConnected(self._client):
            try:
                p.removeAllUserDebugItems()
            except Exception:
                self._clear_debug_lines(self._force_viz_line_ids)
        else:
            self._force_viz_line_ids.clear()
        self._force_viz_line_ids.clear()
        self._force_viz_text_id = -1

    def _clear_wind_visualization(self) -> None:
        self._clear_force_visualization()

    def _clear_thrust_visualization(self) -> None:
        pass

    def _replace_debug_line(
        self,
        line_ids: list[int],
        idx: int,
        start: tuple[float, float, float],
        end: tuple[float, float, float],
        color: list[float],
        width: float = 2.0,
    ) -> None:
        replace_id = line_ids[idx] if idx < len(line_ids) else -1
        uid = p.addUserDebugLine(start, end, color, lineWidth=width, replaceItemUniqueId=replace_id)
        if idx < len(line_ids):
            line_ids[idx] = uid
        else:
            line_ids.append(uid)

    def _draw_force_arrow(
        self,
        line_ids: list[int],
        idx: int,
        origin: tuple[float, float, float],
        force: tuple[float, float, float],
        *,
        color: list[float],
        length_per_n: float,
        min_length: float,
        max_length: float,
        width: float = 2.5,
        draw_head: bool = True,
    ) -> int:
        magnitude = math.sqrt(force[0] ** 2 + force[1] ** 2 + force[2] ** 2)
        if magnitude <= 1e-4:
            calm_end = (origin[0], origin[1], origin[2] + min_length * 0.5)
            self._replace_debug_line(line_ids, idx, origin, calm_end, [0.35, 0.35, 0.35], width=1.2)
            return idx + 1

        direction = self._vec_unit(force)
        length = max(min_length, min(max_length, magnitude * length_per_n))
        tip = self._vec_add(origin, self._vec_scale(direction, length))
        self._replace_debug_line(line_ids, idx, origin, tip, color, width=width)
        idx += 1

        if not draw_head:
            return idx

        ref = (0.0, 0.0, 1.0) if abs(direction[2]) < 0.9 else (1.0, 0.0, 0.0)
        side = self._vec_unit(self._vec_cross(direction, ref))
        head_back = self._vec_scale(direction, -0.22 * length)
        head_side = self._vec_scale(side, 0.12 * length)
        self._replace_debug_line(
            line_ids,
            idx,
            tip,
            self._vec_add(self._vec_add(tip, head_back), head_side),
            color,
            width=width - 0.5,
        )
        idx += 1
        self._replace_debug_line(
            line_ids,
            idx,
            tip,
            self._vec_add(self._vec_add(tip, head_back), self._vec_scale(head_side, -1.0)),
            color,
            width=width - 0.5,
        )
        return idx + 1

    def _drag_force_vector(
        self,
        rel: tuple[float, float, float],
        *,
        k_lin: float | None = None,
        k_quad: float | None = None,
    ) -> tuple[float, float, float]:
        wcfg = self.cfg.wind
        k_lin = wcfg.drag_coeff if k_lin is None else k_lin
        k_quad = wcfg.quad_drag_coeff if k_quad is None else k_quad
        return tuple(self._drag_force_axis(rel[i], k_lin, k_quad) for i in range(3))

    @staticmethod
    def _force_magnitude(force: tuple[float, float, float]) -> float:
        return math.sqrt(force[0] ** 2 + force[1] ** 2 + force[2] ** 2)

    def _update_force_visualization(
        self,
        anchor: tuple[float, float, float],
        pos: tuple[float, float, float],
        orn: tuple[float, float, float, float],
        lin_vel: tuple[float, float, float],
    ) -> None:
        """Draw drag, thrust, and weight arrows on one Newton scale."""
        show_wind = self.cfg.wind.enabled and self.cfg.show_wind_visualization and self.cfg.wind.visualize
        show_thrust = self.cfg.show_thrust_visualization
        if not self.cfg.gui or (not show_wind and not show_thrust):
            return
        if self._client is None or not p.isConnected(self._client):
            return
        stride = max(1, self.cfg.force_viz_update_stride)
        if self._step_count % stride != 0:
            return

        line_ids = self._force_viz_line_ids
        idx = 0
        length_per_n, min_len, max_len = self._force_viz_scale(self.cfg)
        drone_origin = (anchor[0], anchor[1], anchor[2] + 0.35)
        label_parts: list[str] = []

        if show_wind and self._last_wind_info is not None:
            wcfg = self.cfg.wind
            mean_wind = self._last_wind_info["mean_velocity"]
            rel_mean = (
                mean_wind[0] - lin_vel[0],
                mean_wind[1] - lin_vel[1],
                mean_wind[2] - lin_vel[2],
            )
            rel_eff = self._last_wind_info["relative_air_velocity"]
            mean_drag = self._drag_force_vector(rel_mean)
            eff_drag = self._drag_force_vector(rel_eff)
            applied_drag = self._last_wind_info["force"]
            wind_raw = self.cfg.force_viz_wind_mode == "wind_ms"
            if wind_raw:
                # World-frame m/s (matches training obs + episode sample from .env), not air-relative.
                world_eff = self._last_wind_info["velocity"]
                mean_arrow = mean_wind
                eff_arrow = world_eff
                field_arrow = wcfg.velocity
                wind_length_per_unit, wind_min_len, wind_max_len = self._wind_viz_scale(self.cfg)
                mean_mag = self._force_magnitude(mean_arrow)
                eff_mag = self._force_magnitude(eff_arrow)
            else:
                mean_arrow = mean_drag
                eff_arrow = eff_drag
                wind_length_per_unit, wind_min_len, wind_max_len = length_per_n, min_len, max_len
                mean_mag = self._force_magnitude(mean_drag)
                eff_mag = self._force_magnitude(eff_drag)

            idx = self._draw_force_arrow(
                line_ids,
                idx,
                drone_origin,
                mean_arrow,
                color=[0.15, 0.75, 1.0],
                length_per_n=wind_length_per_unit,
                min_length=wind_min_len,
                max_length=wind_max_len,
                width=2.2,
            )
            idx = self._draw_force_arrow(
                line_ids,
                idx,
                (drone_origin[0], drone_origin[1], drone_origin[2] + 0.08),
                eff_arrow,
                color=self._force_magnitude_color(eff_mag, ref_n=1.0 if wind_raw else 3.0),
                length_per_n=wind_length_per_unit,
                min_length=wind_min_len,
                max_length=wind_max_len,
                width=2.8,
            )
            if wcfg.viz_show_applied_drag and not wind_raw:
                applied_mag = self._force_magnitude(applied_drag)
                idx = self._draw_force_arrow(
                    line_ids,
                    idx,
                    (drone_origin[0], drone_origin[1], drone_origin[2] - 0.08),
                    applied_drag,
                    color=[0.95, 0.55, 0.15],
                    length_per_n=length_per_n,
                    min_length=min_len,
                    max_length=max_len,
                    width=2.0,
                )
                label_parts.append(f"drag(appl) {applied_mag:.2f} N")

            half = wcfg.viz_grid_half_extent
            spacing = max(0.25, wcfg.viz_grid_spacing)
            z = anchor[2] + 0.15
            gx = int(round((2 * half) / spacing))
            for ix in range(gx + 1):
                for iy in range(gx + 1):
                    ox = anchor[0] - half + ix * spacing
                    oy = anchor[1] - half + iy * spacing
                    idx = self._draw_force_arrow(
                        line_ids,
                        idx,
                        (ox, oy, z),
                        field_arrow if wind_raw else mean_arrow,
                        color=[0.15, 0.75, 1.0],
                        length_per_n=wind_length_per_unit,
                        min_length=wind_min_len,
                        max_length=wind_max_len,
                        width=1.5,
                        draw_head=False,
                    )

            if wind_raw:
                sample_mag = self._force_magnitude(field_arrow)
                label_parts.append(f"wind(sample) {sample_mag:.3f} m/s")
                label_parts.append(f"wind(mean) {mean_mag:.3f} m/s")
                label_parts.append(f"wind(eff) {eff_mag:.3f} m/s")
            else:
                label_parts.append(f"drag(mean) {mean_mag:.2f} N")
                label_parts.append(f"drag(eff) {eff_mag:.2f} N")

        if show_thrust:
            thrust_dir = self._last_thrust_dir
            thrusts = self._last_thrusts
            total_thrust_n = sum(thrusts)
            total_force = scale_vec(thrust_dir, total_thrust_n)
            thrust_anchor = (
                (drone_origin[0] + 0.14, drone_origin[1], drone_origin[2])
                if show_wind and self._last_wind_info is not None
                else drone_origin
            )
            idx = self._draw_force_arrow(
                line_ids,
                idx,
                thrust_anchor,
                total_force,
                color=[0.05, 0.95, 0.15],
                length_per_n=length_per_n,
                min_length=min_len,
                max_length=max_len,
                width=3.2,
            )
            max_thrust = max(1e-6, self._max_motor)
            for corner_body, thrust in zip(self._corners_body, thrusts):
                origin = world_from_body(pos, orn, corner_body)
                force = scale_vec(thrust_dir, thrust)
                t = max(0.0, min(1.0, thrust / max_thrust))
                color = [0.08 + 0.55 * t, 0.75 + 0.2 * t, 0.08]
                idx = self._draw_force_arrow(
                    line_ids,
                    idx,
                    origin,
                    force,
                    color=color,
                    length_per_n=length_per_n,
                    min_length=min_len,
                    max_length=max_len,
                    width=2.0,
                )
            label_parts.append(f"thrust Σ {total_thrust_n:.2f} N")

        if (show_thrust or show_wind) and self.cfg.wind.viz_show_weight:
            weight = (0.0, 0.0, -MASS * 9.81)
            idx = self._draw_force_arrow(
                line_ids,
                idx,
                (drone_origin[0] + 0.12, drone_origin[1], drone_origin[2]),
                weight,
                color=[0.65, 0.35, 0.95],
                length_per_n=length_per_n,
                min_length=min_len,
                max_length=max_len,
                width=2.0,
            )
            label_parts.append(f"weight {MASS * 9.81:.2f} N")

        if not label_parts:
            label = "force viz (wind m/s)" if self.cfg.force_viz_wind_mode == "wind_ms" else "force viz (N)"
        else:
            label = "  ".join(label_parts)
        text_pos = (anchor[0], anchor[1], anchor[2] + 0.55)
        self._force_viz_text_id = p.addUserDebugText(
            label,
            textPosition=text_pos,
            textColorRGB=[1.0, 1.0, 1.0],
            textSize=1.2,
            replaceItemUniqueId=self._force_viz_text_id,
        )

        while len(line_ids) > idx:
            try:
                p.removeUserDebugItem(line_ids.pop())
            except Exception:
                pass

    def _apply_motor_thrusts(
        self,
        pos: tuple[float, float, float],
        orn: tuple[float, float, float, float],
        thrusts: list[float],
    ) -> None:
        assert self._drone is not None
        thrust_dir = body_z_in_world(orn)
        for corner_body, thrust in zip(self._corners_body, thrusts):
            world_point = world_from_body(pos, orn, corner_body)
            force = scale_vec(thrust_dir, thrust)
            p.applyExternalForce(self._drone, -1, force, world_point, p.WORLD_FRAME)
        self._last_thrusts = list(thrusts)
        self._last_thrust_dir = thrust_dir

    @property
    def episode_step(self) -> int:
        return self._step_count

    @property
    def observation_size(self) -> int:
        wind = (0.0, 0.0, 0.0) if self.cfg.wind.enabled and self.cfg.wind.include_in_obs else None
        return len(
            DroneState(
                step=0,
                pos=(0, 0, 0),
                lin_vel=(0, 0, 0),
                roll=0,
                pitch=0,
                yaw=0,
                ang_vel=(0, 0, 0),
                uprightness=1,
                target=(0, 0, 0),
                err_xy=0,
                err_z=0,
                err_roll=0,
                err_pitch=0,
                wind_world=wind,
            ).as_vector()
        )

    def connect(self) -> None:
        if self._client is not None:
            return
        mode = p.GUI if self.cfg.gui else p.DIRECT
        self._client = p.connect(mode)
        p.setGravity(0.0, 0.0, -9.81)
        load_ground_plane()
        p.setTimeStep(DT)
        self._spawn_drone()

    def close(self) -> None:
        self._clear_force_visualization()
        if self._client is not None and p.isConnected(self._client):
            p.disconnect(self._client)
        self._client = None
        self._drone = None

    def _spawn_drone(self) -> None:
        hx, hy, hz = HALF_EXTENTS
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=HALF_EXTENTS)
        vis = create_drone_visual_shape()
        start = (self.cfg.target_xy[0], self.cfg.target_xy[1], 0.55)
        start_orn = (0.0, 0.0, 0.0, 1.0)
        self._drone = p.createMultiBody(
            baseMass=MASS,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=start,
            baseOrientation=start_orn,
        )
        p.changeDynamics(self._drone, -1, linearDamping=0.05, angularDamping=0.08)
        p.changeDynamics(self._drone, -1, activationState=p.ACTIVATION_STATE_DISABLE_SLEEPING)
        self._corners_body = motor_corners_local(hx, hy, hz)
        self._max_motor = MASS * 9.81 * self.cfg.max_motor_scale

    def reset(
        self,
        *,
        pos: tuple[float, float, float] | None = None,
        orn: tuple[float, float, float, float] | None = None,
        seed: int | None = None,
    ) -> DroneState:
        self.connect()
        assert self._drone is not None

        if seed is not None:
            self._episode_rng = np.random.default_rng(seed)

        if pos is None:
            pos = (self.cfg.target_xy[0], self.cfg.target_xy[1], self.cfg.target_z)
        if orn is None:
            orn = (0.0, 0.0, 0.0, 1.0)

        p.resetBasePositionAndOrientation(self._drone, pos, orn)
        p.resetBaseVelocity(self._drone, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        self._step_count = 0
        self._roll_trim = 0.0
        self._pitch_trim = 0.0
        self._resample_episode_wind()
        self._clear_force_visualization()
        return self._read_state()

    def step(
        self,
        action: list[float] | tuple[float, ...] | None = None,
    ) -> tuple[DroneState, float, bool, dict[str, Any]]:
        """Advance one simulation frame.

        ``action`` (future RL):
            - ``None``: built-in hover PD controller (baseline).
            - 4 floats: per-motor thrust (N), clipped to motor limits.
        """
        assert self._drone is not None
        self._apply_forces(action)
        p.stepSimulation()
        self._step_count += 1
        if self.cfg.step_sleep_s > 0.0:
            time.sleep(self.cfg.step_sleep_s)

        state = self._read_state()
        done, terminal_reason = self._check_termination(state)
        reward_bd = self._compute_reward(state, done, terminal_reason)
        info = {
            "reward_breakdown": reward_bd,
            "terminal_reason": terminal_reason,
            "step": self._step_count,
            "action_mode": "autopilot" if action is None else "motor_thrusts",
        }
        if self._last_wind_info is not None:
            info["wind"] = self._last_wind_info
        return state, reward_bd.total, done, info

    def _read_state(self) -> DroneState:
        assert self._drone is not None
        pos, orn = p.getBasePositionAndOrientation(self._drone)
        lin_vel, ang_vel = p.getBaseVelocity(self._drone)
        roll, pitch, yaw = p.getEulerFromQuaternion(orn)
        thrust_dir = body_z_in_world(orn)
        uprightness = max(-1.0, min(1.0, thrust_dir[2]))

        tx, ty, tz = self.cfg.target_xy[0], self.cfg.target_xy[1], self.cfg.target_z
        err_xy = math.hypot(pos[0] - tx, pos[1] - ty)
        err_z = tz - pos[2]
        err_roll = roll - self._roll_trim
        err_pitch = pitch - self._pitch_trim

        wind_obs = None
        if self.cfg.wind.enabled and self.cfg.wind.include_in_obs:
            wind_obs = self._effective_wind_velocity(self._step_count * DT)

        return DroneState(
            step=self._step_count,
            pos=pos,
            lin_vel=lin_vel,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            ang_vel=ang_vel,
            uprightness=uprightness,
            target=(tx, ty, tz),
            err_xy=err_xy,
            err_z=err_z,
            err_roll=err_roll,
            err_pitch=err_pitch,
            wind_world=wind_obs,
        )

    def _update_wind_turbulence(self) -> None:
        w = self.cfg.wind
        if not any(s > 0.0 for s in w.turbulence_std):
            self._wind_turbulence = (0.0, 0.0, 0.0)
            return
        tau = max(w.turbulence_tau_s, DT)
        alpha = math.exp(-DT / tau)
        beta = math.sqrt(max(0.0, 1.0 - alpha * alpha))
        self._wind_turbulence = tuple(
            alpha * self._wind_turbulence[i] + beta * self._wind_rng.gauss(0.0, w.turbulence_std[i])
            for i in range(3)
        )

    def _mean_wind_velocity(self, sim_time_s: float) -> tuple[float, float, float]:
        """Episode mean wind + sinusoidal gust (no turbulence)."""
        w = self.cfg.wind
        wx, wy, wz = w.velocity
        if w.gust_amplitude:
            gust = w.gust_amplitude * math.sin(2.0 * math.pi * w.gust_freq_hz * sim_time_s)
            wx += gust
            wz += w.vertical_gust_coupling * gust
        return (wx, wy, wz)

    def _effective_wind_velocity(self, sim_time_s: float) -> tuple[float, float, float]:
        """Mean + gust + current turbulence sample (what physics/obs use)."""
        mx, my, mz = self._mean_wind_velocity(sim_time_s)
        tx, ty, tz = self._wind_turbulence
        return (mx + tx, my + ty, mz + tz)

    def _wind_velocity_world(self, sim_time_s: float) -> tuple[float, float, float]:
        """Mean + gust + turbulent component (full 3D). Updates turbulence once."""
        self._update_wind_turbulence()
        return self._effective_wind_velocity(sim_time_s)

    @staticmethod
    def _drag_force_axis(rel: float, k_lin: float, k_quad: float) -> float:
        return k_lin * rel + k_quad * abs(rel) * rel

    def _apply_wind(
        self,
        pos: tuple[float, float, float],
        orn: tuple[float, float, float, float],
        lin_vel: tuple[float, float, float],
    ) -> None:
        """Drag force from air moving relative to the drone (applied each physics step)."""
        wcfg = self.cfg.wind
        if not wcfg.enabled:
            self._last_wind_info = None
            return

        assert self._drone is not None
        sim_time_s = self._step_count * DT
        self._update_wind_turbulence()
        mean_wind = self._mean_wind_velocity(sim_time_s)
        wind = self._effective_wind_velocity(sim_time_s)
        rel = (wind[0] - lin_vel[0], wind[1] - lin_vel[1], wind[2] - lin_vel[2])
        k_lin, k_quad = wcfg.drag_coeff, wcfg.quad_drag_coeff
        force = tuple(self._drag_force_axis(rel[i], k_lin, k_quad) for i in range(3))
        fn = wcfg.force_noise_std
        fn_z = fn * wcfg.force_noise_z_scale
        force = (
            force[0] + self._wind_rng.gauss(0.0, fn),
            force[1] + self._wind_rng.gauss(0.0, fn),
            force[2] + self._wind_rng.gauss(0.0, fn_z),
        )
        cfn = wcfg.corner_force_noise_std
        cfn_z = cfn * wcfg.corner_force_noise_z_scale
        corner_forces: list[tuple[float, float, float]] = []
        for _ in self._corners_body:
            corner_forces.append(
                (
                    force[0] / 4.0 + self._wind_rng.gauss(0.0, cfn),
                    force[1] / 4.0 + self._wind_rng.gauss(0.0, cfn),
                    force[2] / 4.0 + self._wind_rng.gauss(0.0, cfn_z),
                )
            )
        for corner_body, cf in zip(self._corners_body, corner_forces):
            world_point = world_from_body(pos, orn, corner_body)
            p.applyExternalForce(self._drone, -1, cf, world_point, p.WORLD_FRAME)

        tns = wcfg.torque_noise_std
        if tns > 0.0:
            torque = (
                self._wind_rng.gauss(0.0, tns),
                self._wind_rng.gauss(0.0, tns),
                self._wind_rng.gauss(0.0, tns),
            )
            p.applyExternalTorque(self._drone, -1, torque, p.WORLD_FRAME)
        else:
            torque = (0.0, 0.0, 0.0)

        self._last_wind_info = {
            "velocity": wind,
            "mean_velocity": mean_wind,
            "turbulence": self._wind_turbulence,
            "relative_air_velocity": rel,
            "force": force,
            "torque": torque,
        }

    def _apply_forces(
        self,
        action: list[float] | tuple[float, ...] | None,
        *,
        update_viz: bool = True,
    ) -> None:
        assert self._drone is not None
        pos, orn = p.getBasePositionAndOrientation(self._drone)
        lin_vel, ang_vel = p.getBaseVelocity(self._drone)
        roll, pitch, _ = p.getEulerFromQuaternion(orn)

        if self.cfg.hover_balance_thrust:
            mg4 = MASS * 9.81 / 4.0
            thrusts = [mg4, mg4, mg4, mg4]
        elif action is None:
            z_e = self.cfg.target_z - pos[2]
            thrust_sum = MASS * 9.81 + self.cfg.kp_z * z_e - self.cfg.kd_z * lin_vel[2]
            thrust_sum = max(0.0, thrust_sum)
            # Roll: minus sign matches motor layout (+Y / -Y). Pitch needs opposite sign
            # for front/back (+X / -X) so corrections oppose tilt (stable damping).
            mix_r = -self.cfg.kp_rp * (roll - self._roll_trim) - self.cfg.kd_rp * ang_vel[0]
            mix_p = +self.cfg.kp_rp * (pitch - self._pitch_trim) + self.cfg.kd_rp * ang_vel[1]
            thrusts = allocate_motor_thrusts(thrust_sum, mix_r, mix_p, self._signs_xy, self._max_motor)
        else:
            thrusts = [min(self._max_motor, max(0.0, float(t))) for t in action[:4]]
            if len(thrusts) < 4:
                thrusts.extend([0.0] * (4 - len(thrusts)))

        self._apply_motor_thrusts(pos, orn, thrusts)

        self._apply_wind(pos, orn, lin_vel)
        if update_viz:
            self._update_force_visualization(pos, pos, orn, lin_vel)

    def _check_termination(self, state: DroneState) -> tuple[bool, str | None]:
        if abs(state.roll) >= self.cfg.flip_angle_rad or abs(state.pitch) >= self.cfg.flip_angle_rad:
            return True, "flip"
        if state.pos[2] <= self.cfg.crash_z:
            return True, "crash"
        if state.err_xy >= self.cfg.max_xy:
            return True, "out_of_bounds"
        if not self.cfg.unlimited_episode and self._step_count >= self.cfg.max_episode_steps:
            return True, "time_limit"
        return False, None

    def _compute_reward(
        self,
        state: DroneState,
        done: bool,
        terminal_reason: str | None,
    ) -> RewardBreakdown:
        cfg = self.cfg
        alive = cfg.w_alive
        pos_cost = cfg.w_pos_xy * state.err_xy + cfg.w_pos_z * abs(state.err_z)
        att_cost = cfg.w_attitude * (abs(state.err_roll) + abs(state.err_pitch))
        vel_cost = cfg.w_lin_vel * math.sqrt(sum(v * v for v in state.lin_vel)) + cfg.w_ang_vel * math.sqrt(
            sum(w * w for w in state.ang_vel)
        )
        upright_bonus = cfg.w_upright_bonus * max(0.0, state.uprightness)

        position = -pos_cost
        attitude = -att_cost
        velocity = -vel_cost
        terminal_penalty = 0.0
        if done and terminal_reason in ("flip", "crash", "out_of_bounds"):
            terminal_penalty = {
                "flip": -cfg.penalty_flip,
                "crash": -cfg.penalty_crash,
                "out_of_bounds": -cfg.penalty_oob,
            }.get(terminal_reason, 0.0)

        total = alive + position + attitude + velocity + upright_bonus + terminal_penalty
        return RewardBreakdown(
            total=total,
            alive=alive,
            position=position,
            attitude=attitude,
            velocity=velocity,
            upright_bonus=upright_bonus,
            terminal_penalty=terminal_penalty,
            terminal_reason=terminal_reason if terminal_penalty else None,
        )
