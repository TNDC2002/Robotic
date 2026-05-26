"""
Demo: how an RL agent would interact with ``QuadHoverEnv``.

No learning — each frame we call ``env.step()``, then print what the
environment is telling the agent (state, reward parts, done reason).

Run (after activating .venv):
  python rl_interact_demo.py
  python rl_interact_demo.py --gui
  python rl_interact_demo.py --disturb 120   # knock at step 120
"""

from __future__ import annotations

import argparse
import math

from quad_drone_sim import DT, GUI_STEP_SLEEP_S, REALTIME_STEP_SLEEP_S
from quad_hover_env import (
    DroneState,
    EnvConfig,
    QuadHoverEnv,
    RewardBreakdown,
    WindConfig,
    episode_steps_for_seconds,
)


def _deg(rad: float) -> float:
    return math.degrees(rad)


def _status_line(state: DroneState) -> str:
    tilt = max(abs(state.roll), abs(state.pitch))
    if state.uprightness > 0.9 and tilt < math.radians(15):
        mood = "stable hover"
    elif state.uprightness > 0.5:
        mood = "wobbly — agent should correct attitude"
    else:
        mood = "danger — close to flip"
    z_msg = "at target height" if abs(state.err_z) < 0.05 else (
        "below target" if state.err_z > 0 else "above target"
    )
    return f"{mood}; {z_msg}; xy drift {state.err_xy:.3f} m"


def _print_state(state: DroneState) -> None:
    print("  OBSERVATION (what the agent sees):")
    for label, value in zip(state.labels(), state.as_vector()):
        if "rad" in label and "error" not in label:
            extra = f"  ({_deg(value):+.1f} deg)"
        elif label in ("roll error (rad)", "pitch error (rad)"):
            extra = f"  ({_deg(value):+.1f} deg)"
        else:
            extra = ""
        print(f"    {label:18s} {value:+.4f}{extra}")
    print(f"  INTERPRETATION: {_status_line(state)}")


def _print_wind(info: dict) -> None:
    w = info.get("wind")
    if not w:
        return
    vel = w["velocity"]
    force = w["force"]
    turb = w.get("turbulence", (0.0, 0.0, 0.0))
    torque = w.get("torque", (0.0, 0.0, 0.0))
    print("  WIND:")
    print(f"    effective velocity   ({vel[0]:+.3f}, {vel[1]:+.3f}, {vel[2]:+.3f}) m/s")
    print(f"    turbulence (OU)      ({turb[0]:+.3f}, {turb[1]:+.3f}, {turb[2]:+.3f}) m/s")
    print(f"    drag force (total)   ({force[0]:+.3f}, {force[1]:+.3f}, {force[2]:+.3f}) N")
    print(f"    torque noise         ({torque[0]:+.3f}, {torque[1]:+.3f}, {torque[2]:+.3f}) N·m")


def _print_reward(bd: RewardBreakdown) -> None:
    print("  REWARD (what the agent is encouraged to do):")
    print(f"    {bd.explain()}")
    print("    Meaning:")
    print("      + alive        : small bonus each frame you survive")
    print("      + position     : closer to target (x,y,z) is better")
    print("      + attitude     : level roll/pitch is better")
    print("      + velocity     : slow motion is better")
    print("      + upright_bonus: body 'up' aligned with world up")
    if bd.terminal_penalty:
        print(f"      + terminal     : big penalty for {bd.terminal_reason}")


def _print_step_header(step: int, done: bool) -> None:
    bar = "=" * 60
    print(bar)
    print(f"FRAME {step}" + ("  [EPISODE DONE]" if done else ""))


def run_episode(
    *,
    gui: bool,
    max_steps: int | None,
    episode_seconds: float | None,
    print_every: int,
    disturb_at: int | None,
    step_sleep_s: float,
    wind: WindConfig | None,
) -> None:
    cfg = EnvConfig(gui=gui, step_sleep_s=step_sleep_s)
    if wind is not None:
        cfg.wind = wind
    if episode_seconds is not None:
        cfg.max_episode_steps = episode_steps_for_seconds(episode_seconds)
    elif max_steps is not None:
        cfg.max_episode_steps = max_steps
    # else: keep EnvConfig.max_episode_steps default from quad_hover_env.py

    horizon = cfg.max_episode_steps
    env = QuadHoverEnv(cfg)

    print("QuadHoverEnv — RL interaction demo (no learning)")
    print(f"  Episode horizon: {horizon} steps (~{cfg.episode_duration_s:.1f} s at 240 Hz)")
    if step_sleep_s > 0.0:
        print(f"  Playback: {step_sleep_s * 1000:.1f} ms/frame (~{1.0 / step_sleep_s:.0f} FPS view)")
    else:
        print("  Playback: max speed (no frame sleep)")
    if cfg.wind.enabled:
        w = cfg.wind
        print(
            f"  Wind: v={w.velocity} m/s, drag={w.drag_coeff}, quad_drag={w.quad_drag_coeff}, "
            f"turbulence_std={w.turbulence_std}, gust={w.gust_amplitude} m/s"
        )
    else:
        print("  Wind: off")
    print(f"  Task: hold near target {cfg.target_xy + (cfg.target_z,)} and avoid flip/crash")
    print(f"  Observation size: {env.observation_size} floats")
    print(f"  Action: None each step -> built-in hover autopilot (placeholder for your policy)")
    if disturb_at is not None:
        print(f"  Disturbance at step {disturb_at}: random motor thrust for 30 frames")
    print()

    state = env.reset()
    cumulative_reward = 0.0
    disturb_left = 0

    _print_step_header(state.step, False)
    _print_state(state)
    print("  (initial state after reset — no reward yet)")
    print()

    for _ in range(horizon):
        action = None
        if disturb_at is not None and env.episode_step >= disturb_at and disturb_left < 30:
            # Simulate a bad policy briefly so reward/termination are visible.
            action = [0.3, 0.3, 2.0, 2.0]
            disturb_left += 1

        state, reward, done, info = env.step(action)
        cumulative_reward += reward
        bd = info["reward_breakdown"]

        if state.step == 0 or state.step % print_every == 0 or done:
            _print_step_header(state.step, done)
            _print_state(state)
            _print_wind(info)
            _print_reward(bd)
            print(f"  done={done}  terminal_reason={info['terminal_reason']!r}")
            print(f"  cumulative_reward={cumulative_reward:+.2f}")
            print()

        if done:
            reason = info["terminal_reason"]
            if reason == "time_limit":
                print("Episode ended: survived full horizon — good baseline.")
            elif reason == "flip":
                print("Episode ended: FLIP — roll/pitch exceeded safe angle.")
            elif reason == "crash":
                print("Episode ended: CRASH — altitude too low.")
            elif reason == "out_of_bounds":
                print("Episode ended: drifted too far in XY.")
            break

    env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Print QuadHoverEnv state/reward each frame.")
    parser.add_argument("--gui", action="store_true", help="PyBullet GUI window")
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Max simulation frames (overrides EnvConfig default if set)",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=None,
        help="Episode length in seconds (converted to steps at 240 Hz; overrides --steps)",
    )
    parser.add_argument("--print-every", type=int, default=60, help="Print every N frames")
    parser.add_argument(
        "--disturb",
        type=int,
        default=None,
        metavar="STEP",
        help="Apply a short bad motor command at this step",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=None,
        metavar="SEC",
        help=f"Pause each frame in seconds (default {GUI_STEP_SLEEP_S} with --gui, else 0)",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help=f"Playback at sim rate (~{1.0 / DT:.0f} Hz); overrides --sleep",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="No frame sleep (overrides --sleep / --gui default)",
    )
    parser.add_argument(
        "--wind",
        nargs=3,
        type=float,
        metavar=("WX", "WY", "WZ"),
        default=None,
        help="Enable wind with constant world velocity (m/s), e.g. --wind 0.8 0 0",
    )
    parser.add_argument(
        "--wind-drag",
        type=float,
        default=0.4,
        help="Linear drag coefficient when --wind is set (default 0.4)",
    )
    parser.add_argument(
        "--gust",
        type=float,
        default=0.0,
        help="Sinusoidal gust amplitude added to wind.x (m/s)",
    )
    parser.add_argument(
        "--wind-turbulence",
        nargs="?",
        const=1.0,
        default=None,
        type=float,
        metavar="SCALE",
        help="Enable 3D turbulent wind; optional scale (default 1.0). Use 0 to disable.",
    )
    parser.add_argument(
        "--wind-noise",
        type=float,
        default=None,
        help="Extra random drag force std (N per axis); default from WindConfig",
    )
    parser.add_argument(
        "--wind-seed",
        type=int,
        default=None,
        help="RNG seed for reproducible wind noise",
    )
    args = parser.parse_args()
    if args.steps is not None and args.seconds is not None:
        parser.error("Use only one of --steps or --seconds")
    if args.fast and args.realtime:
        parser.error("Use only one of --fast or --realtime")

    if args.fast:
        step_sleep_s = 0.0
    elif args.realtime:
        step_sleep_s = REALTIME_STEP_SLEEP_S
    elif args.sleep is not None:
        step_sleep_s = max(0.0, args.sleep)
    elif args.gui:
        step_sleep_s = GUI_STEP_SLEEP_S
    else:
        step_sleep_s = 0.0

    wind_cfg = None
    if args.wind is not None:
        turb_scale = 1.0 if args.wind_turbulence is None else max(0.0, args.wind_turbulence)
        base_turb = (0.4, 0.4, 0.3)
        if args.wind_turbulence == 0.0:
            turb = (0.0, 0.0, 0.0)
        else:
            turb = tuple(turb_scale * t for t in base_turb)
        wind_cfg = WindConfig(
            enabled=True,
            velocity=(args.wind[0], args.wind[1], args.wind[2]),
            drag_coeff=args.wind_drag,
            gust_amplitude=args.gust,
            turbulence_std=turb,
            seed=args.wind_seed,
        )
        if args.wind_noise is not None:
            wind_cfg.force_noise_std = args.wind_noise

    run_episode(
        gui=args.gui,
        max_steps=args.steps,
        episode_seconds=args.seconds,
        print_every=args.print_every,
        disturb_at=args.disturb,
        step_sleep_s=step_sleep_s,
        wind=wind_cfg,
    )


if __name__ == "__main__":
    main()
