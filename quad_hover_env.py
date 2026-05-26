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
from typing import Any

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
    # Smoothed 3D turbulence (Ornstein–Uhlenbeck); affects X, Y, and Z wind components.
    turbulence_std: tuple[float, float, float] = (0.4, 0.4, 0.3)
    turbulence_tau_s: float = 0.3  # correlation time; lower = jerkier gusts
    force_noise_std: float = 0.12  # extra random force (N) per axis on total drag
    corner_force_noise_std: float = 0.08  # random force (N) per axis at each motor corner
    torque_noise_std: float = 0.04  # random torque (N·m) per axis
    seed: int | None = None  # None = nondeterministic turbulence
    include_in_obs: bool = True  # append effective wind_xyz to observation when enabled


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
    # Wall-clock pause after each stepSimulation (0 = as fast as possible).
    step_sleep_s: float = 0.0
    wind: WindConfig = field(default_factory=WindConfig)
    # PD hover controller (used when action is None)
    kp_z: float = 18.0
    kd_z: float = 4.5
    kp_rp: float = 0.45
    kd_rp: float = 0.08
    max_motor_scale: float = 0.65  # fraction of mg per motor cap
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
    ) -> DroneState:
        self.connect()
        assert self._drone is not None

        if pos is None:
            pos = (self.cfg.target_xy[0], self.cfg.target_xy[1], self.cfg.target_z)
        if orn is None:
            orn = (0.0, 0.0, 0.0, 1.0)

        p.resetBasePositionAndOrientation(self._drone, pos, orn)
        p.resetBaseVelocity(self._drone, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        self._step_count = 0
        self._roll_trim = 0.0
        self._pitch_trim = 0.0
        self._wind_turbulence = (0.0, 0.0, 0.0)
        self._wind_rng = random.Random(self.cfg.wind.seed)
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
            wind_obs = self._wind_velocity_world(self._step_count * DT)

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

    def _wind_velocity_world(self, sim_time_s: float) -> tuple[float, float, float]:
        """Mean + gust + turbulent component (full 3D)."""
        w = self.cfg.wind
        self._update_wind_turbulence()
        wx, wy, wz = w.velocity
        if w.gust_amplitude:
            gust = w.gust_amplitude * math.sin(2.0 * math.pi * w.gust_freq_hz * sim_time_s)
            wx += gust
            # Couple a fraction of gust into vertical component (updraft/downdraft).
            wz += 0.35 * gust
        tx, ty, tz = self._wind_turbulence
        return (wx + tx, wy + ty, wz + tz)

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
        wind = self._wind_velocity_world(sim_time_s)
        rel = (wind[0] - lin_vel[0], wind[1] - lin_vel[1], wind[2] - lin_vel[2])
        k_lin, k_quad = wcfg.drag_coeff, wcfg.quad_drag_coeff
        force = tuple(self._drag_force_axis(rel[i], k_lin, k_quad) for i in range(3))
        fn = wcfg.force_noise_std
        force = tuple(
            force[i] + self._wind_rng.gauss(0.0, fn) for i in range(3)
        )
        cfn = wcfg.corner_force_noise_std
        corner_forces: list[tuple[float, float, float]] = []
        for _ in self._corners_body:
            corner_forces.append(
                (
                    force[0] / 4.0 + self._wind_rng.gauss(0.0, cfn),
                    force[1] / 4.0 + self._wind_rng.gauss(0.0, cfn),
                    force[2] / 4.0 + self._wind_rng.gauss(0.0, cfn),
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
            "turbulence": self._wind_turbulence,
            "relative_air_velocity": rel,
            "force": force,
            "torque": torque,
        }

    def _apply_forces(self, action: list[float] | tuple[float, ...] | None) -> None:
        assert self._drone is not None
        pos, orn = p.getBasePositionAndOrientation(self._drone)
        lin_vel, ang_vel = p.getBaseVelocity(self._drone)
        roll, pitch, _ = p.getEulerFromQuaternion(orn)

        if action is None:
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

        thrust_dir = body_z_in_world(orn)
        for corner_body, thrust in zip(self._corners_body, thrusts):
            world_point = world_from_body(pos, orn, corner_body)
            force = scale_vec(thrust_dir, thrust)
            p.applyExternalForce(self._drone, -1, force, world_point, p.WORLD_FRAME)

        self._apply_wind(pos, orn, lin_vel)

    def _check_termination(self, state: DroneState) -> tuple[bool, str | None]:
        if abs(state.roll) >= self.cfg.flip_angle_rad or abs(state.pitch) >= self.cfg.flip_angle_rad:
            return True, "flip"
        if state.pos[2] <= self.cfg.crash_z:
            return True, "crash"
        if state.err_xy >= self.cfg.max_xy:
            return True, "out_of_bounds"
        if self._step_count >= self.cfg.max_episode_steps:
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
