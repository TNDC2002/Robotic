"""
Load environment settings from ``.env`` and process environment variables.

Priority (highest first): explicit function/CLI arguments → environment variables →
dataclass defaults in ``quad_nav_env`` / ``quad_hover_env``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

from quad_drone_sim import GUI_STEP_SLEEP_S, MASS, REALTIME_STEP_SLEEP_S
from quad_hover_env import WindConfig
from quad_nav_env import NavEnvConfig, NavWindRandomizationConfig, sync_max_motor_scale

_REPO_ROOT = Path(__file__).resolve().parent
load_dotenv(_REPO_ROOT / ".env", override=False)

_GRAVITY = 9.81
_BASE_TURBULENCE = (0.4, 0.4, 0.06)


def hover_thrust_per_motor_n() -> float:
    """Per-motor thrust (N) that balances gravity when level: total thrust = m·g."""
    return MASS * _GRAVITY / 4.0


def _parse_float(name: str, default: float | None = None) -> float | None:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


def _parse_float_range(
    min_key: str,
    max_key: str,
    default_min: float,
    default_max: float,
) -> tuple[float, float]:
    lo = _parse_float(min_key, default_min)
    hi = _parse_float(max_key, default_max)
    assert lo is not None and hi is not None
    return (lo, hi)


def _parse_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _parse_int(name: str, default: int | None = None) -> int | None:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


@dataclass(frozen=True)
class MotorThrustSettings:
    min_n: float
    max_n: float

    def __post_init__(self) -> None:
        if self.min_n < 0.0:
            raise ValueError(f"motor thrust min must be >= 0, got {self.min_n}")
        if self.max_n <= self.min_n:
            raise ValueError(
                f"motor thrust max must be > min ({self.min_n}), got {self.max_n}"
            )


def resolve_gui_step_sleep_s(
    *,
    gui: bool,
    sleep: float | None = None,
    realtime: bool = False,
    fast: bool = False,
    frame_skip: int = 1,
    default_gui_sleep: float | None = None,
) -> float:
    """Playback delay once per env step (RL step for nav; physics step for hover demo)."""
    if not gui:
        return 0.0
    if fast or _parse_bool("GUI_FAST"):
        return 0.0
    if realtime or _parse_bool("GUI_REALTIME"):
        return REALTIME_STEP_SLEEP_S * max(1, frame_skip)
    if sleep is not None:
        return max(0.0, sleep)
    env_sleep = _parse_float("GUI_STEP_SLEEP_S")
    if env_sleep is not None:
        return max(0.0, env_sleep)
    fallback = GUI_STEP_SLEEP_S if default_gui_sleep is None else default_gui_sleep
    return max(0.0, fallback)


def load_hover_balance_mode(*, cli_flag: bool | None = None) -> bool:
    if cli_flag is not None:
        return cli_flag
    return _parse_bool("HOVER_BALANCE_MODE", False)


def load_wind_config(
    *,
    enabled: bool | None = None,
    velocity: tuple[float, float, float] | None = None,
) -> WindConfig:
    """Fixed wind field from ``WIND_*`` env vars (used in demo / non-randomized runs)."""
    default = WindConfig()
    turb_scale = _parse_float("WIND_TURBULENCE_SCALE", 1.0) or 0.0
    base_x = _parse_float("WIND_TURBULENCE_BASE_X", _BASE_TURBULENCE[0]) or 0.0
    base_y = _parse_float("WIND_TURBULENCE_BASE_Y", _BASE_TURBULENCE[1]) or 0.0
    base_z = _parse_float("WIND_TURBULENCE_BASE_Z", _BASE_TURBULENCE[2]) or 0.0
    turb = (
        turb_scale * base_x,
        turb_scale * base_y,
        turb_scale * base_z,
    )
    if enabled is None:
        enabled = _parse_bool("WIND_ENABLED", default.enabled)
    if velocity is None:
        velocity = (
            _parse_float("WIND_VX", default.velocity[0]) or 0.0,
            _parse_float("WIND_VY", default.velocity[1]) or 0.0,
            _parse_float("WIND_VZ", default.velocity[2]) or 0.0,
        )
    seed_raw = os.getenv("WIND_SEED")
    seed = default.seed if seed_raw is None or not seed_raw.strip() else int(seed_raw)
    return WindConfig(
        enabled=enabled,
        velocity=velocity,
        drag_coeff=_parse_float("WIND_DRAG", default.drag_coeff) or default.drag_coeff,
        quad_drag_coeff=_parse_float("WIND_QUAD_DRAG", default.quad_drag_coeff) or default.quad_drag_coeff,
        gust_amplitude=_parse_float("WIND_GUST", default.gust_amplitude) or 0.0,
        gust_freq_hz=_parse_float("WIND_GUST_FREQ_HZ", default.gust_freq_hz) or default.gust_freq_hz,
        vertical_gust_coupling=_parse_float("WIND_VERTICAL_GUST_COUPLING", default.vertical_gust_coupling)
        or default.vertical_gust_coupling,
        turbulence_std=turb,
        turbulence_tau_s=_parse_float("WIND_TURBULENCE_TAU_S", default.turbulence_tau_s)
        or default.turbulence_tau_s,
        force_noise_std=_parse_float("WIND_FORCE_NOISE", default.force_noise_std) or default.force_noise_std,
        force_noise_z_scale=_parse_float("WIND_FORCE_NOISE_Z_SCALE", default.force_noise_z_scale)
        or default.force_noise_z_scale,
        corner_force_noise_std=_parse_float("WIND_CORNER_NOISE", default.corner_force_noise_std)
        or default.corner_force_noise_std,
        corner_force_noise_z_scale=_parse_float("WIND_CORNER_NOISE_Z_SCALE", default.corner_force_noise_z_scale)
        or default.corner_force_noise_z_scale,
        torque_noise_std=_parse_float("WIND_TORQUE_NOISE", default.torque_noise_std) or default.torque_noise_std,
        seed=seed,
        include_in_obs=_parse_bool("WIND_INCLUDE_IN_OBS", default.include_in_obs),
    )


def load_wind_randomization_config(*, enabled: bool | None = None) -> NavWindRandomizationConfig:
    """Per-episode wind randomization ranges for navigation training."""
    default = NavWindRandomizationConfig()
    if enabled is None:
        enabled = _parse_bool("WIND_RANDOMIZATION", default.enabled)
    return NavWindRandomizationConfig(
        enabled=enabled,
        speed_range=_parse_float_range("WIND_SPEED_MIN", "WIND_SPEED_MAX", *default.speed_range),
        drag_range=_parse_float_range("WIND_DRAG_MIN", "WIND_DRAG_MAX", *default.drag_range),
        quad_drag_range=_parse_float_range(
            "WIND_QUAD_DRAG_MIN", "WIND_QUAD_DRAG_MAX", *default.quad_drag_range
        ),
        turbulence_scale_range=_parse_float_range(
            "WIND_TURBULENCE_SCALE_MIN", "WIND_TURBULENCE_SCALE_MAX", *default.turbulence_scale_range
        ),
        force_noise_range=_parse_float_range(
            "WIND_FORCE_NOISE_MIN", "WIND_FORCE_NOISE_MAX", *default.force_noise_range
        ),
        corner_noise_range=_parse_float_range(
            "WIND_CORNER_NOISE_MIN", "WIND_CORNER_NOISE_MAX", *default.corner_noise_range
        ),
        torque_noise_range=_parse_float_range(
            "WIND_TORQUE_NOISE_MIN", "WIND_TORQUE_NOISE_MAX", *default.torque_noise_range
        ),
        gust_amplitude_range=_parse_float_range("WIND_GUST_MIN", "WIND_GUST_MAX", *default.gust_amplitude_range),
        base_turbulence_std=(
            _parse_float("WIND_TURBULENCE_BASE_X", default.base_turbulence_std[0]) or 0.0,
            _parse_float("WIND_TURBULENCE_BASE_Y", default.base_turbulence_std[1]) or 0.0,
            _parse_float("WIND_TURBULENCE_BASE_Z", default.base_turbulence_std[2]) or 0.0,
        ),
        vertical_gust_coupling=_parse_float("WIND_VERTICAL_GUST_COUPLING", default.vertical_gust_coupling)
        or default.vertical_gust_coupling,
        force_noise_z_scale=_parse_float("WIND_FORCE_NOISE_Z_SCALE", default.force_noise_z_scale)
        or default.force_noise_z_scale,
        corner_force_noise_z_scale=_parse_float("WIND_CORNER_NOISE_Z_SCALE", default.corner_force_noise_z_scale)
        or default.corner_force_noise_z_scale,
    )


def load_motor_thrust_settings(
    *,
    min_n: float | None = None,
    max_n: float | None = None,
    default_min_n: float = 0.0,
    default_max_n: float = 8.0,
) -> MotorThrustSettings:
    resolved_min = min_n if min_n is not None else _parse_float("MOTOR_THRUST_MIN_N", default_min_n)
    resolved_max = max_n if max_n is not None else _parse_float("MOTOR_THRUST_MAX_N", default_max_n)
    assert resolved_min is not None and resolved_max is not None
    return MotorThrustSettings(resolved_min, resolved_max)


def build_hover_env_config(
    *,
    gui: bool = False,
    step_sleep_s: float | None = None,
    hover_balance: bool | None = None,
    wind: WindConfig | None = None,
    episode_seconds: float | None = None,
) -> "EnvConfig":
    from quad_hover_env import EnvConfig, episode_steps_for_seconds

    resolved_sleep = resolve_gui_step_sleep_s(gui=gui, sleep=step_sleep_s, frame_skip=1)
    cfg = EnvConfig(
        gui=gui,
        step_sleep_s=resolved_sleep,
        hover_balance_thrust=load_hover_balance_mode(cli_flag=hover_balance),
        wind=wind if wind is not None else load_wind_config(),
    )
    if cfg.hover_balance_thrust:
        cfg.max_xy = max(cfg.max_xy, 12.0)
    if episode_seconds is not None:
        cfg.max_episode_steps = episode_steps_for_seconds(episode_seconds)
    return cfg


def build_nav_env_config(
    *,
    gui: bool = False,
    step_sleep_s: float | None = None,
    no_wind: bool = False,
    action_mode: Literal["motors", "mixer"] = "motors",
    motor_thrust_min: float | None = None,
    motor_thrust_max: float | None = None,
    gui_realtime: bool = False,
    gui_fast: bool = False,
    frame_skip: int | None = None,
) -> NavEnvConfig:
    thrust = load_motor_thrust_settings(
        min_n=motor_thrust_min,
        max_n=motor_thrust_max,
        default_min_n=NavEnvConfig.motor_thrust_min,
        default_max_n=NavEnvConfig.motor_thrust_max,
    )
    skip = NavEnvConfig.frame_skip if frame_skip is None else frame_skip
    wr = load_wind_randomization_config(enabled=not no_wind)
    resolved_sleep = resolve_gui_step_sleep_s(
        gui=gui,
        sleep=step_sleep_s,
        realtime=gui_realtime,
        fast=gui_fast,
        frame_skip=skip,
    )
    cfg = NavEnvConfig(
        gui=gui,
        step_sleep_s=resolved_sleep,
        wind_randomization=wr,
        action_mode=action_mode,
        motor_thrust_min=thrust.min_n,
        motor_thrust_max=thrust.max_n,
    )
    if not wr.enabled:
        cfg.wind = load_wind_config()
    sync_max_motor_scale(cfg)
    return cfg
