"""
Load environment settings from ``.env`` and process environment variables.

Priority (highest first): explicit CLI/function arguments (when provided) →
**``.env``** (always overrides shell/conda env) → code defaults.
"""

from __future__ import annotations

import project_env  # noqa: F401 — load .env before any os.getenv below

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Union

from quad_drone_sim import GUI_STEP_SLEEP_S, MASS, REALTIME_STEP_SLEEP_S
from quad_nav_env import NavEnvConfig, NavMapConfig, sync_max_motor_scale
from wind_settings import describe_wind_settings, load_wind_settings

_REPO_ROOT = Path(__file__).resolve().parent

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


def _env_float(name: str, default: float) -> float:
    """Read float env var; ``0`` is valid (unlike ``parsed or default``)."""
    parsed = _parse_float(name)
    return default if parsed is None else parsed


def _env_int(name: str, default: int) -> int:
    """Read int env var; ``0`` is valid (unlike ``parsed or default``)."""
    parsed = _parse_int(name)
    return default if parsed is None else parsed


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip()


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
    """Max tilt from vertical (degrees): angle between body +Z and world +Z."""
    if deg is not None:
        return deg
    parsed = _env_float("SAFE_ATTITUDE_DEG", default_deg)
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


def load_nav_end_on_unsafe_attitude(*, cli_flag: bool | None = None) -> bool:
    if cli_flag is not None:
        return cli_flag
    return _parse_bool("NAV_END_ON_UNSAFE_ATTITUDE", False)


def load_penalty_unsafe_attitude_end(
    *,
    penalty: float | None = None,
    default: float | None = None,
) -> float:
    d = NavEnvConfig()
    fallback = d.penalty_unsafe_attitude_end if default is None else default
    if penalty is not None:
        return penalty
    return _env_float("PENALTY_UNSAFE_ATTITUDE_END", fallback)


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
    resolved_min = min_n if min_n is not None else _env_float("MOTOR_THRUST_MIN_N", default_min_n)
    resolved_max = max_n if max_n is not None else _env_float("MOTOR_THRUST_MAX_N", default_max_n)
    return MotorThrustSettings(resolved_min, resolved_max)


@dataclass(frozen=True)
class NavSpawnSettings:
    """Navigation spawn altitude and goal Z sampling bounds."""

    spawn_z: float
    goal_z_max_rise: float
    map_z_max: float

    def __post_init__(self) -> None:
        if self.spawn_z <= 0.0:
            raise ValueError(f"NAV_SPAWN_Z must be > 0, got {self.spawn_z}")
        if self.goal_z_max_rise < 0.0:
            raise ValueError(f"NAV_GOAL_Z_MAX_RISE must be >= 0, got {self.goal_z_max_rise}")
        if self.map_z_max < self.spawn_z:
            raise ValueError(
                f"NAV_MAP_Z_MAX ({self.map_z_max}) must be >= NAV_SPAWN_Z ({self.spawn_z})"
            )


def load_nav_spawn_settings(
    *,
    spawn_z: float | None = None,
    goal_z_max_rise: float | None = None,
    map_z_max: float | None = None,
) -> NavSpawnSettings:
    """Read nav spawn/goal altitude from args, then ``NAV_*`` env vars."""
    d = NavEnvConfig()
    resolved_spawn = spawn_z if spawn_z is not None else _env_float("NAV_SPAWN_Z", d.spawn_z)
    resolved_rise = (
        goal_z_max_rise
        if goal_z_max_rise is not None
        else _env_float("NAV_GOAL_Z_MAX_RISE", d.goal_z_max_rise)
    )
    resolved_z_max = (
        map_z_max if map_z_max is not None else _env_float("NAV_MAP_Z_MAX", d.map.z_max)
    )
    return NavSpawnSettings(
        spawn_z=resolved_spawn,
        goal_z_max_rise=resolved_rise,
        map_z_max=resolved_z_max,
    )


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
    w_safe_attitude: float
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
    w_safe_attitude: float | None = None,
    success_bonus: float | None = None,
    crash_penalty: float | None = None,
    time_limit_penalty: float | None = None,
) -> NavRewardSettings:
    """Read navigation weights from ``REWARD_*`` / ``PENALTY_*`` env vars (see REWARD.md)."""
    d = NavEnvConfig()

    def _reward(name: str, cli: float | None, default: float) -> float:
        if cli is not None:
            return cli
        return _env_float(name, default)

    def _penalty(name: str, cli: float | None, default: float, *, legacy: str | None = None) -> float:
        if cli is not None:
            return cli
        parsed = _parse_float(name)
        if parsed is not None:
            return parsed
        if legacy is not None:
            legacy_val = _parse_float(legacy)
            if legacy_val is not None:
                return legacy_val
        return default

    return NavRewardSettings(
        w_progress=_reward("REWARD_W_PROGRESS", w_progress, d.w_progress),
        w_goal_dist=_penalty("PENALTY_W_GOAL_DIST", w_goal_dist, d.w_goal_dist, legacy="REWARD_W_GOAL_DIST"),
        w_goal_alt=_penalty("PENALTY_W_GOAL_ALT", w_goal_alt, d.w_goal_alt, legacy="REWARD_W_GOAL_ALT"),
        w_goal_alt_progress=_reward(
            "REWARD_W_GOAL_ALT_PROGRESS", w_goal_alt_progress, d.w_goal_alt_progress
        ),
        w_alive=_reward("REWARD_W_ALIVE", w_alive, d.w_alive),
        w_attitude=_penalty("PENALTY_W_ATTITUDE", w_attitude, d.w_attitude, legacy="REWARD_W_ATTITUDE"),
        w_ang_vel=_penalty("PENALTY_W_ANG_VEL", w_ang_vel, d.w_ang_vel, legacy="REWARD_W_ANG_VEL"),
        w_lin_vel=_penalty("PENALTY_W_LIN_VEL", w_lin_vel, d.w_lin_vel, legacy="REWARD_W_LIN_VEL"),
        w_upright=_reward("REWARD_W_UPRIGHT", w_upright, d.w_upright),
        w_action_rate=_penalty(
            "PENALTY_W_ACTION_RATE", w_action_rate, d.w_action_rate, legacy="REWARD_W_ACTION_RATE"
        ),
        w_unsafe_attitude=_penalty(
            "PENALTY_W_UNSAFE_ATTITUDE",
            w_unsafe_attitude,
            d.w_unsafe_attitude,
            legacy="REWARD_W_UNSAFE_ATTITUDE",
        ),
        w_safe_attitude=_reward("REWARD_W_SAFE_ATTITUDE", w_safe_attitude, d.w_safe_attitude),
        success_bonus=_reward("REWARD_SUCCESS_BONUS", success_bonus, d.success_bonus),
        crash_penalty=_penalty("PENALTY_CRASH", crash_penalty, d.crash_penalty, legacy="REWARD_CRASH_PENALTY"),
        time_limit_penalty=_penalty(
            "PENALTY_TIME_LIMIT", time_limit_penalty, d.time_limit_penalty, legacy="REWARD_TIME_LIMIT_PENALTY"
        ),
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
    initial = lr if lr is not None else _env_float("PPO_LEARNING_RATE", default_initial)
    final = lr_final if lr_final is not None else _env_float("PPO_LR_FINAL", default_final)
    sched_raw = (
        schedule if schedule is not None else _env_str("PPO_LR_SCHEDULE", default_schedule)
    ).strip().lower()
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


def resolve_model_zip_path(path: str) -> Path:
    """Normalize a SB3 ``.zip`` model path and verify the file exists."""
    p = Path(path.strip())
    if p.suffix.lower() != ".zip":
        p = Path(f"{p}.zip")
    if not p.is_file():
        raise FileNotFoundError(p)
    return p


def load_ppo_resume_model(*, resume: str | None = None) -> str | None:
    """Optional checkpoint path from ``PPO_RESUME_MODEL`` or CLI ``--resume``."""
    return load_ppo_resume_settings(resume=resume).model_path


@dataclass(frozen=True)
class PpoResumeSettings:
    """Resume training: load policy weights; timestep counter always resets."""

    model_path: str | None
    load_optimizer: bool


def load_ppo_resume_settings(
    *,
    resume: str | None = None,
    load_optimizer: bool | None = None,
) -> PpoResumeSettings:
    """Read resume path and optimizer flag from CLI / ``PPO_RESUME_*`` env vars."""
    if resume is not None:
        raw = resume.strip()
        path = None if not raw else raw
    else:
        raw = os.getenv("PPO_RESUME_MODEL")
        path = None if raw is None or not str(raw).strip() else str(raw).strip()
    return PpoResumeSettings(
        model_path=path,
        load_optimizer=(
            load_optimizer
            if load_optimizer is not None
            else _parse_bool("PPO_RESUME_LOAD_OPTIMIZER", False)
        ),
    )


def describe_ppo_resume_settings(settings: PpoResumeSettings) -> str:
    if settings.model_path is None:
        return "disabled (train from scratch)"
    opt = "weights + optimizer" if settings.load_optimizer else "weights only (fresh optimizer)"
    return f"{settings.model_path} ({opt}, timesteps reset)"


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
            else _env_int("PPO_CKPT_FREQ", default_ckpt_freq)
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


@dataclass(frozen=True)
class PpoEvalSettings:
    """Periodic evaluation during PPO training."""

    eval_freq: int
    n_eval_episodes: int

    def __post_init__(self) -> None:
        if self.eval_freq < 1:
            raise ValueError(f"PPO_EVAL_FREQ must be >= 1, got {self.eval_freq}")
        if self.n_eval_episodes < 1:
            raise ValueError(f"PPO_EVAL_EPISODES must be >= 1, got {self.n_eval_episodes}")


def load_ppo_eval_settings(
    *,
    eval_freq: int | None = None,
    n_eval_episodes: int | None = None,
    default_eval_freq: int = 20_000,
    default_n_eval_episodes: int = 5,
) -> PpoEvalSettings:
    """Read eval cadence from CLI args, then ``PPO_EVAL_*`` env vars."""
    return PpoEvalSettings(
        eval_freq=(
            eval_freq
            if eval_freq is not None
            else _env_int("PPO_EVAL_FREQ", default_eval_freq)
        ),
        n_eval_episodes=(
            n_eval_episodes
            if n_eval_episodes is not None
            else _env_int("PPO_EVAL_EPISODES", default_n_eval_episodes)
        ),
    )


@dataclass(frozen=True)
class PpoEarlyStopSettings:
    """Stop PPO training when eval mean reward stops improving."""

    enabled: bool
    patience: int
    min_delta: float
    min_evals: int

    def __post_init__(self) -> None:
        if self.patience < 1:
            raise ValueError(f"PPO_EARLY_STOP_PATIENCE must be >= 1, got {self.patience}")
        if self.min_evals < 1:
            raise ValueError(f"PPO_EARLY_STOP_MIN_EVALS must be >= 1, got {self.min_evals}")
        if self.min_delta < 0.0:
            raise ValueError(f"PPO_EARLY_STOP_MIN_DELTA must be >= 0, got {self.min_delta}")


def load_ppo_early_stop_settings(
    *,
    early_stop: bool | None = None,
    patience: int | None = None,
    min_delta: float | None = None,
    min_evals: int | None = None,
    default_enabled: bool = False,
    default_patience: int = 5,
    default_min_delta: float = 0.0,
    default_min_evals: int = 3,
) -> PpoEarlyStopSettings:
    """Read early stopping from CLI args, then ``PPO_EARLY_STOP*`` env vars."""
    return PpoEarlyStopSettings(
        enabled=(
            early_stop
            if early_stop is not None
            else _parse_bool("PPO_EARLY_STOP", default_enabled)
        ),
        patience=(
            patience
            if patience is not None
            else _env_int("PPO_EARLY_STOP_PATIENCE", default_patience)
        ),
        min_delta=(
            min_delta
            if min_delta is not None
            else _env_float("PPO_EARLY_STOP_MIN_DELTA", default_min_delta)
        ),
        min_evals=(
            min_evals
            if min_evals is not None
            else _env_int("PPO_EARLY_STOP_MIN_EVALS", default_min_evals)
        ),
    )


def describe_ppo_early_stop_settings(settings: PpoEarlyStopSettings) -> str:
    if not settings.enabled:
        return "disabled"
    return (
        f"enabled (patience={settings.patience} evals, "
        f"min_delta={settings.min_delta}, min_evals={settings.min_evals})"
    )


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
    spawn = load_nav_spawn_settings()
    nav_map = NavMapConfig(cruise_z=spawn.spawn_z, z_max=spawn.map_z_max)
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
        map=nav_map,
        spawn_z=spawn.spawn_z,
        goal_z_max_rise=spawn.goal_z_max_rise,
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
        w_safe_attitude=rewards.w_safe_attitude,
        success_bonus=rewards.success_bonus,
        crash_penalty=rewards.crash_penalty,
        time_limit_penalty=rewards.time_limit_penalty,
        end_on_unsafe_attitude=load_nav_end_on_unsafe_attitude(),
        penalty_unsafe_attitude_end=load_penalty_unsafe_attitude_end(),
    )
    if cfg.hover_balance_thrust:
        cfg.max_xy = max(cfg.max_xy, 12.0)
    if episode_seconds is not None:
        cfg.max_episode_steps = episode_steps_for_seconds(episode_seconds)
    sync_max_motor_scale(cfg)
    return cfg
