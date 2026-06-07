"""
Wind configuration: per-episode sampling from ``WIND_*_MIN`` / ``WIND_*_MAX`` ranges in ``.env``.

Used by navigation (train/eval), hover demo, and inference. Set ``WIND_ENABLED=0`` or pass
``no_wind=True`` to disable wind entirely.
"""

from __future__ import annotations

import project_env  # noqa: F401 — load .env before os.getenv below

import math
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from quad_hover_env import WindConfig

# OU turbulence base (m/s); scaled by sampled WIND_TURBULENCE_SCALE_* each episode.
_BASE_TURBULENCE_STD = (0.4, 0.4, 0.06)


def _parse_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


def _parse_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _parse_range(min_key: str, max_key: str, default_min: float, default_max: float) -> tuple[float, float]:
    lo = _parse_float(min_key, default_min)
    hi = _parse_float(max_key, default_max)
    if hi < lo:
        raise ValueError(f"{max_key} must be >= {min_key} ({hi} < {lo})")
    return (lo, hi)


@dataclass
class WindRandomizationConfig:
    """Uniform sampling ranges — one draw per episode at reset."""

    speed_range: tuple[float, float] = (0.5, 2.5)
    drag_range: tuple[float, float] = (0.35, 0.75)
    quad_drag_range: tuple[float, float] = (0.03, 0.08)
    turbulence_scale_range: tuple[float, float] = (1.0, 3.0)
    force_noise_range: tuple[float, float] = (0.08, 0.5)
    corner_noise_range: tuple[float, float] = (0.04, 0.15)
    torque_noise_range: tuple[float, float] = (0.03, 0.12)
    gust_amplitude_range: tuple[float, float] = (0.0, 0.5)
    # Fixed physics (not range-sampled); tuned in code.
    base_turbulence_std: tuple[float, float, float] = _BASE_TURBULENCE_STD
    turbulence_tau_s: float = 0.3
    gust_freq_hz: float = 0.15
    vertical_gust_coupling: float = 0.10
    force_noise_z_scale: float = 0.20
    corner_force_noise_z_scale: float = 0.20


@dataclass(frozen=True)
class WindSettings:
    """Master wind switch + sampling ranges (+ optional episode seed)."""

    enabled: bool
    ranges: WindRandomizationConfig
    include_in_obs: bool
    seed: int | None = None


def load_wind_randomization_config() -> WindRandomizationConfig:
    """Load ``WIND_*_MIN`` / ``WIND_*_MAX`` ranges from environment."""
    d = WindRandomizationConfig()
    return WindRandomizationConfig(
        speed_range=_parse_range("WIND_SPEED_MIN", "WIND_SPEED_MAX", *d.speed_range),
        drag_range=_parse_range("WIND_DRAG_MIN", "WIND_DRAG_MAX", *d.drag_range),
        quad_drag_range=_parse_range("WIND_QUAD_DRAG_MIN", "WIND_QUAD_DRAG_MAX", *d.quad_drag_range),
        turbulence_scale_range=_parse_range(
            "WIND_TURBULENCE_SCALE_MIN", "WIND_TURBULENCE_SCALE_MAX", *d.turbulence_scale_range
        ),
        force_noise_range=_parse_range("WIND_FORCE_NOISE_MIN", "WIND_FORCE_NOISE_MAX", *d.force_noise_range),
        corner_noise_range=_parse_range("WIND_CORNER_NOISE_MIN", "WIND_CORNER_NOISE_MAX", *d.corner_noise_range),
        torque_noise_range=_parse_range("WIND_TORQUE_NOISE_MIN", "WIND_TORQUE_NOISE_MAX", *d.torque_noise_range),
        gust_amplitude_range=_parse_range("WIND_GUST_MIN", "WIND_GUST_MAX", *d.gust_amplitude_range),
    )


def load_wind_settings(*, enabled: bool | None = None, no_wind: bool = False) -> WindSettings:
    """Read ``WIND_ENABLED``, ranges, ``WIND_INCLUDE_IN_OBS``, optional ``WIND_SEED``."""
    if no_wind:
        enabled = False
    elif enabled is None:
        enabled = _parse_bool("WIND_ENABLED", True)
    seed_raw = os.getenv("WIND_SEED")
    seed = None if seed_raw is None or not seed_raw.strip() else int(seed_raw)
    return WindSettings(
        enabled=enabled,
        ranges=load_wind_randomization_config(),
        include_in_obs=_parse_bool("WIND_INCLUDE_IN_OBS", True),
        seed=seed,
    )


def sample_episode_wind(rng: np.random.Generator | Any, settings: WindSettings) -> WindConfig:
    """Draw one ``WindConfig`` for an episode (random horizontal direction + uniform ranges)."""
    if not settings.enabled:
        return WindConfig(enabled=False, include_in_obs=False)

    wr = settings.ranges
    speed = float(rng.uniform(*wr.speed_range))
    angle = float(rng.uniform(0.0, 2.0 * math.pi))
    turb_scale = float(rng.uniform(*wr.turbulence_scale_range))
    turb = tuple(turb_scale * t for t in wr.base_turbulence_std)
    episode_seed = int(rng.integers(0, 2**31 - 1))
    return WindConfig(
        enabled=True,
        velocity=(speed * math.cos(angle), speed * math.sin(angle), 0.0),
        drag_coeff=float(rng.uniform(*wr.drag_range)),
        quad_drag_coeff=float(rng.uniform(*wr.quad_drag_range)),
        turbulence_std=turb,
        turbulence_tau_s=wr.turbulence_tau_s,
        force_noise_std=float(rng.uniform(*wr.force_noise_range)),
        force_noise_z_scale=wr.force_noise_z_scale,
        corner_force_noise_std=float(rng.uniform(*wr.corner_noise_range)),
        corner_force_noise_z_scale=wr.corner_force_noise_z_scale,
        torque_noise_std=float(rng.uniform(*wr.torque_noise_range)),
        gust_amplitude=float(rng.uniform(*wr.gust_amplitude_range)),
        gust_freq_hz=wr.gust_freq_hz,
        vertical_gust_coupling=wr.vertical_gust_coupling,
        include_in_obs=settings.include_in_obs,
        seed=episode_seed,
    )


def describe_sampled_wind(wind: WindConfig) -> str:
    """One-line summary of the wind draw for this episode."""
    v = wind.velocity
    speed = math.hypot(v[0], v[1])
    angle_deg = math.degrees(math.atan2(v[1], v[0]))
    return (
        f"sampled episode wind: speed={speed:.3f} m/s @ {angle_deg:+.0f}°, "
        f"v=({v[0]:+.3f}, {v[1]:+.3f}, {v[2]:+.3f}), drag={wind.drag_coeff:.3f}, "
        f"quad_drag={wind.quad_drag_coeff:.4f}, turb={wind.turbulence_std}, "
        f"gust={wind.gust_amplitude:.3f}"
    )


def describe_wind_settings(settings: WindSettings, *, cli_no_wind: bool = False) -> str:
    if cli_no_wind or not settings.enabled:
        if cli_no_wind and settings.enabled:
            return "off (--no-wind)"
        return "off (WIND_ENABLED=0)"
    wr = settings.ranges
    return (
        f"random per episode — speed {wr.speed_range[0]:g}–{wr.speed_range[1]:g} m/s, "
        f"drag {wr.drag_range[0]:g}–{wr.drag_range[1]:g}, "
        f"turbulence scale {wr.turbulence_scale_range[0]:g}–{wr.turbulence_scale_range[1]:g}"
    )
