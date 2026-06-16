"""
Wind-tunnel / hover-balance demo using the same ``QuadNavEnv`` and wind sampling as PPO train/eval.

Each episode draws wind from ``WIND_*_MIN/MAX`` in ``.env`` (identical to ``train_nav_rl.py``).

Run:
  python rl_interact_demo.py --gui --hover-balance
  python rl_interact_demo.py --gui --hover-balance --wind-viz-raw
  python rl_interact_demo.py --gui --hover-balance --wind-viz-raw --wind-viz-scale 50
  python rl_interact_demo.py --gui --hover-balance --wind-viz-raw --seed 0
"""

from __future__ import annotations

import argparse

import numpy as np

from quad_drone_sim import DT, GUI_STEP_SLEEP_S, REALTIME_STEP_SLEEP_S
from quad_hover_env import episode_steps_for_seconds
from quad_nav_env import NavEnvConfig, QuadNavEnv


def _print_wind(info: dict) -> None:
    w = info.get("wind")
    if not w:
        return
    if isinstance(w, dict) and "velocity" in w and "force" not in w:
        vel = w["velocity"]
        print("  WIND (episode mean):")
        print(f"    velocity             ({vel[0]:+.3f}, {vel[1]:+.3f}, {vel[2]:+.3f}) m/s")
        print(f"    drag_coeff           {w.get('drag_coeff', 0):.3f}")
        return
    vel = w["velocity"]
    force = w["force"]
    turb = w.get("turbulence", (0.0, 0.0, 0.0))
    torque = w.get("torque", (0.0, 0.0, 0.0))
    print("  WIND (live):")
    print(f"    effective velocity   ({vel[0]:+.3f}, {vel[1]:+.3f}, {vel[2]:+.3f}) m/s")
    print(f"    turbulence (OU)      ({turb[0]:+.3f}, {turb[1]:+.3f}, {turb[2]:+.3f}) m/s")
    print(f"    drag force (total)   ({force[0]:+.3f}, {force[1]:+.3f}, {force[2]:+.3f}) N")
    print(f"    torque noise         ({torque[0]:+.3f}, {torque[1]:+.3f}, {torque[2]:+.3f}) N·m")


def _print_step_header(step: int, done: bool) -> None:
    print("=" * 60)
    print(f"STEP {step}" + ("  [EPISODE DONE]" if done else ""))


def run_episode(
    *,
    gui: bool,
    max_steps: int | None,
    episode_seconds: float | None,
    print_every: int,
    step_sleep_s: float,
    no_wind: bool = False,
    hover_balance: bool = True,
    wind_viz_raw: bool = False,
    wind_viz_scale: float | None = None,
    seed: int | None = None,
) -> None:
    from env_config import build_nav_env_config, hover_thrust_per_motor_n
    from wind_settings import describe_sampled_wind, describe_wind_settings

    cfg = build_nav_env_config(
        gui=gui,
        step_sleep_s=step_sleep_s,
        no_wind=no_wind,
        hover_balance=hover_balance,
        gui_unlimited=True,
        episode_seconds=episode_seconds,
        wind_viz_raw=wind_viz_raw,
        wind_viz_scale=wind_viz_scale,
    )
    if max_steps is not None:
        cfg.max_episode_steps = max_steps

    env = QuadNavEnv(cfg)
    nav: NavEnvConfig = cfg

    print("QuadNavEnv wind preview — same env + wind sampling as train/eval")
    if cfg.unlimited_episode:
        print("  Episode horizon: unlimited (GUI_UNLIMITED_EPISODE — until crash or goal)")
    else:
        rl_steps = cfg.max_episode_steps // max(1, nav.frame_skip)
        print(
            f"  Episode horizon: {cfg.max_episode_steps} physics steps "
            f"(~{rl_steps} RL steps, frame_skip={nav.frame_skip})"
        )
    if step_sleep_s > 0.0:
        print(f"  Playback: {step_sleep_s * 1000:.1f} ms/RL step")
    else:
        print("  Playback: max speed (no frame sleep)")
    print(f"  Wind ranges: {describe_wind_settings(cfg.wind_settings, cli_no_wind=no_wind)}")
    if cfg.hover_balance_thrust:
        t = hover_thrust_per_motor_n()
        print(
            f"  Mode: HOVER BALANCE — fixed {t:.3f} N/motor ({4 * t:.3f} N total ≈ m·g); "
            "wind identical to navigation training"
        )
    else:
        print("  Mode: neutral motor action (0) — use --hover-balance for wind-tunnel view")
    if cfg.force_viz_wind_mode == "wind_ms":
        eff_scale = cfg.force_viz_length_per_ms * cfg.force_viz_wind_length_scale
        print(
            f"  Wind viz: episode wind field (m/s), "
            f"scale={eff_scale:g} m per (m/s) "
            f"({cfg.force_viz_length_per_ms:g} × {cfg.force_viz_wind_length_scale:g})"
        )
    else:
        print("  Wind viz: drag force (N)")
    if seed is not None:
        print(f"  Episode seed: {seed} (reproducible wind draw)")
    else:
        print("  Episode seed: random (new wind draw each run)")
    print(f"  Goal (nav task): {env._goal_pos}")
    print(f"  Observation size: {env.observation_size} floats")
    print()

    obs = env.reset(seed=seed)
    if cfg.wind.enabled:
        print(f"  {describe_sampled_wind(cfg.wind)}")
        print()

    _print_step_header(0, False)
    print(f"  spawn={env._spawn_pos}")
    print(f"  obs dim={obs.shape[0]}")
    print()

    cumulative_reward = 0.0
    done = False
    rl_step = 0
    neutral_action = np.zeros(4, dtype=np.float32)

    while not done:
        obs, reward, done, info = env.step(neutral_action)
        cumulative_reward += reward
        rl_step += 1

        if rl_step == 1 or rl_step % print_every == 0 or done:
            _print_step_header(info.get("step", rl_step), done)
            print(f"  dist_to_goal={info.get('distance_to_goal', 0):.3f} m  goal={info.get('goal')}")
            print(f"  unsafe_attitude={info.get('unsafe_attitude', False)}")
            _print_wind(info)
            print(f"  reward={reward:+.3f}  cumulative={cumulative_reward:+.2f}")
            print(f"  done={done}  terminal_reason={info.get('terminal_reason')!r}")
            print()

        if done:
            reason = info.get("terminal_reason")
            if reason == "success":
                print("Episode ended: reached goal.")
            elif reason == "crash":
                print("Episode ended: floor crash.")
            elif reason == "time_limit":
                print("Episode ended: time limit.")
            break

    env.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="QuadNavEnv wind-tunnel demo (same randomized wind as train/eval)."
    )
    parser.add_argument("--gui", action="store_true", help="PyBullet GUI window")
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Max physics steps (overrides default episode length)",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=None,
        help="Episode length in seconds at 240 Hz physics",
    )
    parser.add_argument("--print-every", type=int, default=30, help="Print every N RL steps")
    parser.add_argument(
        "--sleep",
        type=float,
        default=None,
        metavar="SEC",
        help=f"Pause per RL step (default {GUI_STEP_SLEEP_S} with --gui)",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help=f"Playback at sim rate (~{1.0 / (DT * NavEnvConfig.frame_skip):.0f} RL steps/s)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="No frame sleep (overrides --sleep / --gui default)",
    )
    parser.add_argument(
        "--hover-balance",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Fixed m·g/4 per motor (default: HOVER_BALANCE_MODE in .env, else on)",
    )
    parser.add_argument(
        "--no-wind",
        action="store_true",
        help="Disable wind (overrides WIND_ENABLED in .env)",
    )
    parser.add_argument(
        "--wind-viz-raw",
        action="store_true",
        help="Show episode wind field (m/s) from .env sampling; longer arrows than drag (N)",
    )
    parser.add_argument(
        "--wind-viz-scale",
        type=float,
        default=None,
        metavar="N",
        help="Multiply wind arrow length in --wind-viz-raw mode (default: FORCE_VIZ_WIND_LENGTH_SCALE in .env, else 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Episode RNG seed (default: random wind draw from WIND_* ranges each run)",
    )
    args = parser.parse_args()
    if args.steps is not None and args.seconds is not None:
        parser.error("Use only one of --steps or --seconds")
    if args.fast and args.realtime:
        parser.error("Use only one of --fast or --realtime")

    from env_config import load_hover_balance_mode, resolve_gui_step_sleep_s

    hover_balance = args.hover_balance
    if hover_balance is None:
        hover_balance = load_hover_balance_mode(default=True)

    max_steps = args.steps
    if args.seconds is not None:
        max_steps = episode_steps_for_seconds(args.seconds)

    step_sleep_s = resolve_gui_step_sleep_s(
        gui=args.gui,
        sleep=args.sleep,
        realtime=args.realtime,
        fast=args.fast,
        frame_skip=NavEnvConfig.frame_skip,
    )

    run_episode(
        gui=args.gui,
        max_steps=max_steps,
        episode_seconds=None,
        print_every=args.print_every,
        step_sleep_s=step_sleep_s,
        no_wind=args.no_wind,
        hover_balance=hover_balance,
        wind_viz_raw=args.wind_viz_raw,
        wind_viz_scale=args.wind_viz_scale,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
