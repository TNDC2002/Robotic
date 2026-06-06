"""
Load environment settings from ``.env`` and process environment variables.

Priority (highest first): explicit function/CLI arguments → environment variables →
dataclass defaults in ``quad_nav_env`` / ``quad_hover_env``.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Union

from dotenv import load_dotenv

from quad_drone_sim import GUI_STEP_SLEEP_S, MASS, REALTIME_STEP_SLEEP_S
from quad_nav_env import NavEnvConfig, sync_max_motor_scale
from wind_settings import describe_wind_settings, load_wind_settings

_REPO_ROOT = Path(__file__).resolve().parent
load_dotenv(_REPO_ROOT / ".env", override=False)

_GRAVITY = 9.81


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


def load_hover_balance_mode(*, cli_flag: bool | None = None, default: bool = False) -> bool:
    if cli_flag is not None:
        return cli_flag
    return _parse_bool("HOVER_BALANCE_MODE", default)


def load_safe_attitude_deg(*, deg: float | None = None, default_deg: float = 70.0) -> float:
    """Max |roll|/|pitch| (degrees) before nav ``unsafe_attitude`` penalty; hover flip limit."""
    if deg is not None:
        return deg
    parsed = _parse_float("SAFE_ATTITUDE_DEG", default_deg)
    assert parsed is not None
    if not 0.0 < parsed < 90.0:
        raise ValueError(f"SAFE_ATTITUDE_DEG must be in (0, 90), got {parsed}")
    return parsed


def safe_attitude_rad(*, deg: float | None = None) -> float:
    return math.radians(load_safe_attitude_deg(deg=deg))


def load_gui_unlimited_episode(*, gui: bool, cli_flag: bool | None = None) -> bool:
    """When True with ``gui``, episodes run until success/crash (no time_limit)."""
    if not gui:
        return False
    if cli_flag is not None:
        return cli_flag
    return _parse_bool("GUI_UNLIMITED_EPISODE", False)


def describe_nav_wind_settings(cfg: NavEnvConfig, *, cli_no_wind: bool = False) -> str:
    """Alias for :func:`wind_settings.describe_wind_settings` using nav env config."""
    settings = cfg.wind_settings or load_wind_settings()
    return describe_wind_settings(settings, cli_no_wind=cli_no_wind)


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


@dataclass(frozen=True)
class NavRewardSettings:
    """Navigation reward weights (see REWARD.md)."""

    w_progress: float
    w_goal_dist: float
    w_goal_alt: float
    w_goal_alt_progress: float
    w_alive: float
    w_attitude: float
    w_ang_vel: float
    w_lin_vel: float
    w_upright: float
    w_action_rate: float
    w_unsafe_attitude: float
    success_bonus: float
    crash_penalty: float
    time_limit_penalty: float


def load_nav_reward_settings(
    *,
    w_progress: float | None = None,
    w_goal_dist: float | None = None,
    w_goal_alt: float | None = None,
    w_goal_alt_progress: float | None = None,
    w_alive: float | None = None,
    w_attitude: float | None = None,
    w_ang_vel: float | None = None,
    w_lin_vel: float | None = None,
    w_upright: float | None = None,
    w_action_rate: float | None = None,
    w_unsafe_attitude: float | None = None,
    success_bonus: float | None = None,
    crash_penalty: float | None = None,
    time_limit_penalty: float | None = None,
) -> NavRewardSettings:
    """Read navigation reward weights from args, then ``REWARD_*`` env vars."""
    d = NavEnvConfig()

    def _w(name: str, cli: float | None, default: float) -> float:
        if cli is not None:
            return cli
        parsed = _parse_float(name, default)
        assert parsed is not None
        return parsed

    return NavRewardSettings(
        w_progress=_w("REWARD_W_PROGRESS", w_progress, d.w_progress),
        w_goal_dist=_w("REWARD_W_GOAL_DIST", w_goal_dist, d.w_goal_dist),
        w_goal_alt=_w("REWARD_W_GOAL_ALT", w_goal_alt, d.w_goal_alt),
        w_goal_alt_progress=_w(
            "REWARD_W_GOAL_ALT_PROGRESS", w_goal_alt_progress, d.w_goal_alt_progress
        ),
        w_alive=_w("REWARD_W_ALIVE", w_alive, d.w_alive),
        w_attitude=_w("REWARD_W_ATTITUDE", w_attitude, d.w_attitude),
        w_ang_vel=_w("REWARD_W_ANG_VEL", w_ang_vel, d.w_ang_vel),
        w_lin_vel=_w("REWARD_W_LIN_VEL", w_lin_vel, d.w_lin_vel),
        w_upright=_w("REWARD_W_UPRIGHT", w_upright, d.w_upright),
        w_action_rate=_w("REWARD_W_ACTION_RATE", w_action_rate, d.w_action_rate),
        w_unsafe_attitude=_w("REWARD_W_UNSAFE_ATTITUDE", w_unsafe_attitude, d.w_unsafe_attitude),
        success_bonus=_w("REWARD_SUCCESS_BONUS", success_bonus, d.success_bonus),
        crash_penalty=_w("REWARD_CRASH_PENALTY", crash_penalty, d.crash_penalty),
        time_limit_penalty=_w("REWARD_TIME_LIMIT_PENALTY", time_limit_penalty, d.time_limit_penalty),
    )


LrScheduleName = Literal["constant", "linear", "cosine"]
PpoLearningRate = Union[float, Callable[[float], float]]


@dataclass(frozen=True)
class PpoLrSettings:
    """PPO learning rate schedule (Stable-Baselines3 ``progress_remaining`` 1 → 0)."""

    initial: float
    final: float
    schedule: LrScheduleName

    def __post_init__(self) -> None:
        if self.initial <= 0.0 or self.final < 0.0:
            raise ValueError(f"learning rates must be positive (initial={self.initial}, final={self.final})")
        if self.final > self.initial:
            raise ValueError(f"PPO_LR_FINAL must be <= PPO_LEARNING_RATE ({self.final} > {self.initial})")


def load_ppo_lr_settings(
    *,
    lr: float | None = None,
    lr_final: float | None = None,
    schedule: str | None = None,
    default_initial: float = 3e-4,
    default_final: float = 1e-5,
    default_schedule: LrScheduleName = "linear",
) -> PpoLrSettings:
    """Read PPO LR from CLI args, then ``PPO_*`` env vars."""
    initial = lr if lr is not None else (_parse_float("PPO_LEARNING_RATE", default_initial) or default_initial)
    final = lr_final if lr_final is not None else (_parse_float("PPO_LR_FINAL", default_final) or default_final)
    sched_raw = (schedule or os.getenv("PPO_LR_SCHEDULE") or default_schedule).strip().lower()
    if sched_raw not in ("constant", "linear", "cosine"):
        raise ValueError(f"PPO_LR_SCHEDULE must be constant, linear, or cosine (got {sched_raw!r})")
    return PpoLrSettings(initial=initial, final=final, schedule=sched_raw)  # type: ignore[arg-type]


def build_ppo_learning_rate(settings: PpoLrSettings) -> PpoLearningRate:
    """Return a constant float or SB3 schedule callable."""
    if settings.schedule == "constant":
        return settings.initial

    def _lerp(progress_remaining: float) -> float:
        return settings.final + (settings.initial - settings.final) * progress_remaining

    if settings.schedule == "linear":
        return _lerp

    def _cosine(progress_remaining: float) -> float:
        t = 1.0 - progress_remaining
        return settings.final + 0.5 * (settings.initial - settings.final) * (1.0 + math.cos(math.pi * t))

    return _cosine


def describe_ppo_learning_rate(settings: PpoLrSettings) -> str:
    if settings.schedule == "constant":
        return f"constant {settings.initial:g}"
    return f"{settings.schedule} {settings.initial:g} → {settings.final:g}"


@dataclass(frozen=True)
class PpoCheckpointSettings:
    """Which PPO artifacts to write under ``runs/nav_ppo/`` during training."""

    save_ckpt: bool
    ckpt_freq: int
    save_best: bool
    save_final: bool

    def __post_init__(self) -> None:
        if self.ckpt_freq < 1:
            raise ValueError(f"PPO_CKPT_FREQ must be >= 1, got {self.ckpt_freq}")


def load_ppo_checkpoint_settings(
    *,
    save_ckpt: bool | None = None,
    ckpt_freq: int | None = None,
    save_best: bool | None = None,
    save_final: bool | None = None,
    default_ckpt_freq: int = 50_000,
    default_save_ckpt: bool = True,
    default_save_best: bool = True,
    default_save_final: bool = True,
) -> PpoCheckpointSettings:
    """Read checkpoint/save flags from CLI args, then ``PPO_SAVE_*`` / ``PPO_CKPT_FREQ`` env vars."""
    return PpoCheckpointSettings(
        save_ckpt=(
            save_ckpt
            if save_ckpt is not None
            else _parse_bool("PPO_SAVE_CKPT", default_save_ckpt)
        ),
        ckpt_freq=(
            ckpt_freq
            if ckpt_freq is not None
            else (_parse_int("PPO_CKPT_FREQ", default_ckpt_freq) or default_ckpt_freq)
        ),
        save_best=(
            save_best
            if save_best is not None
            else _parse_bool("PPO_SAVE_BEST", default_save_best)
        ),
        save_final=(
            save_final
            if save_final is not None
            else _parse_bool("PPO_SAVE_FINAL", default_save_final)
        ),
    )


def describe_ppo_checkpoint_settings(settings: PpoCheckpointSettings) -> str:
    parts: list[str] = []
    if settings.save_ckpt:
        parts.append(f"ckpt/ every {settings.ckpt_freq:,} timesteps")
    if settings.save_best:
        parts.append("best/ on eval improvement")
    if settings.save_final:
        parts.append("final_model at end")
    return ", ".join(parts) if parts else "disabled (no checkpoints)"


def build_hover_env_config(
    *,
    gui: bool = False,
    step_sleep_s: float | None = None,
    hover_balance: bool | None = None,
    no_wind: bool = False,
    episode_seconds: float | None = None,
) -> "EnvConfig":
    from quad_hover_env import EnvConfig, episode_steps_for_seconds

    resolved_sleep = resolve_gui_step_sleep_s(gui=gui, sleep=step_sleep_s, frame_skip=1)
    cfg = EnvConfig(
        gui=gui,
        step_sleep_s=resolved_sleep,
        flip_angle_rad=safe_attitude_rad(),
        unlimited_episode=load_gui_unlimited_episode(gui=gui),
        hover_balance_thrust=load_hover_balance_mode(cli_flag=hover_balance),
        wind_settings=load_wind_settings(no_wind=no_wind),
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
    gui_unlimited: bool | None = None,
    safe_attitude_deg: float | None = None,
    hover_balance: bool | None = None,
    episode_seconds: float | None = None,
) -> NavEnvConfig:
    from quad_hover_env import episode_steps_for_seconds

    thrust = load_motor_thrust_settings(
        min_n=motor_thrust_min,
        max_n=motor_thrust_max,
        default_min_n=NavEnvConfig.motor_thrust_min,
        default_max_n=NavEnvConfig.motor_thrust_max,
    )
    skip = NavEnvConfig.frame_skip if frame_skip is None else frame_skip
    wind_settings = load_wind_settings(no_wind=no_wind)
    resolved_sleep = resolve_gui_step_sleep_s(
        gui=gui,
        sleep=step_sleep_s,
        realtime=gui_realtime,
        fast=gui_fast,
        frame_skip=skip,
    )
    rewards = load_nav_reward_settings()
    cfg = NavEnvConfig(
        gui=gui,
        step_sleep_s=resolved_sleep,
        flip_angle_rad=safe_attitude_rad(deg=safe_attitude_deg),
        unlimited_episode=load_gui_unlimited_episode(gui=gui, cli_flag=gui_unlimited),
        hover_balance_thrust=load_hover_balance_mode(cli_flag=hover_balance),
        wind_settings=wind_settings,
        action_mode=action_mode,
        motor_thrust_min=thrust.min_n,
        motor_thrust_max=thrust.max_n,
        w_progress=rewards.w_progress,
        w_goal_dist=rewards.w_goal_dist,
        w_goal_alt=rewards.w_goal_alt,
        w_goal_alt_progress=rewards.w_goal_alt_progress,
        w_alive=rewards.w_alive,
        w_attitude=rewards.w_attitude,
        w_ang_vel=rewards.w_ang_vel,
        w_lin_vel=rewards.w_lin_vel,
        w_upright=rewards.w_upright,
        w_action_rate=rewards.w_action_rate,
        w_unsafe_attitude=rewards.w_unsafe_attitude,
        success_bonus=rewards.success_bonus,
        crash_penalty=rewards.crash_penalty,
        time_limit_penalty=rewards.time_limit_penalty,
    )
    if cfg.hover_balance_thrust:
        cfg.max_xy = max(cfg.max_xy, 12.0)
    if episode_seconds is not None:
        cfg.max_episode_steps = episode_steps_for_seconds(episode_seconds)
    sync_max_motor_scale(cfg)
    return cfg
