"""
Manual keyboard flight using PPO **mixer** action mode (roll/pitch mix, not 4 raw motors).

Uses ``QuadNavEnv`` with ``action_mode="mixer"`` — same decode path as
``train_nav_rl.py --action-mode mixer``.

Controls (PyBullet GUI window must have focus):
  W / S  — pitch command (nose down / nose up)
  A / D  — roll command (roll left / roll right)
  ESC    — quit

When no tilt keys are held, action is ``[0, 0, 0, 0]``: total thrust ≈ m·g and
zero roll/pitch mix → four motors share hover thrust (drone floats level).

Run:
  python manual_mixer_demo.py --gui
  python manual_mixer_demo.py --gui --no-wind
"""

from __future__ import annotations

import argparse

import numpy as np
import pybullet as p

from env_config import build_nav_env_config, hover_thrust_per_motor_n, resolve_gui_step_sleep_s
from quad_drone_sim import DT
from quad_nav_env import NavEnvConfig, QuadNavEnv


def _key_down(keys: dict[int, int], char: str) -> bool:
    code = ord(char.lower())
    return code in keys and bool(keys[code] & p.KEY_IS_DOWN)


def mixer_action_from_keys(keys: dict[int, int]) -> tuple[np.ndarray, str]:
    """Build mixer action [thrust_delta, roll, pitch, yaw]; neutral = hover balance."""
    roll = 0.0
    pitch = 0.0
    parts: list[str] = []

    if _key_down(keys, "a"):
        roll -= 1.0
        parts.append("roll-")
    if _key_down(keys, "d"):
        roll += 1.0
        parts.append("roll+")
    if _key_down(keys, "w"):
        pitch += 1.0
        parts.append("pitch+")
    if _key_down(keys, "s"):
        pitch -= 1.0
        parts.append("pitch-")

    if not parts:
        return np.zeros(4, dtype=np.float32), "hover (m·g/4 per motor)"

    return np.array([0.0, roll, pitch, 0.0], dtype=np.float32), "+".join(parts)


def run_manual_mixer(
    *,
    gui: bool,
    no_wind: bool,
    step_sleep_s: float,
    mix_strength: float,
) -> None:
    if not gui:
        raise SystemExit("Manual mixer requires --gui (keyboard input is read from PyBullet).")

    cfg = build_nav_env_config(
        gui=True,
        step_sleep_s=step_sleep_s,
        no_wind=no_wind,
        action_mode="mixer",
        hover_balance=False,
        gui_unlimited=True,
    )
    if mix_strength != 1.0:
        cfg.attitude_mix_scale *= mix_strength

    env = QuadNavEnv(cfg)
    hover_n = hover_thrust_per_motor_n()

    print("Manual mixer demo — action_mode=mixer (same as train_nav_rl.py --action-mode mixer)")
    print(f"  Hover neutral: action [0,0,0,0] → ~{hover_n:.3f} N/motor ({4 * hover_n:.3f} N total)")
    print(f"  attitude_mix_scale={cfg.attitude_mix_scale:.3f}")
    print("  Keys: W/S pitch, A/D roll, ESC quit (click the PyBullet window first)")
    print()

    obs = env.reset(seed=0)
    done = False
    rl_step = 0

    while True:
        keys = p.getKeyboardEvents()
        if 27 in keys and keys[27] & p.KEY_WAS_TRIGGERED:
            break

        action, label = mixer_action_from_keys(keys)
        obs, reward, done, info = env.step(action)
        rl_step += 1

        if rl_step == 1 or rl_step % 30 == 0 or done:
            thrusts = info.get("motor_thrusts", [])
            thrust_s = ", ".join(f"{t:.3f}" for t in thrusts) if thrusts else "?"
            print(
                f"step {info.get('step', rl_step):5d}  mode={label:24s}  "
                f"tilt={info.get('tilt_deg', 0):.1f}°  "
                f"thrusts=[{thrust_s}] N  "
                f"dist_goal={info.get('distance_to_goal', 0):.2f} m"
            )

        if done:
            reason = info.get("terminal_reason")
            print(f"  episode end: {reason!r} — resetting")
            obs = env.reset()
            done = False

    env.close()
    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyboard manual flight (mixer action mode)")
    parser.add_argument("--gui", action="store_true", help="PyBullet GUI (required)")
    parser.add_argument("--no-wind", action="store_true", help="Disable wind")
    parser.add_argument(
        "--mix-strength",
        type=float,
        default=1.0,
        help="Scale attitude_mix_scale for roll/pitch keys (default 1.0)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=None,
        help="Pause per RL step (default 0.01 s with --gui)",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help=f"Match sim rate (~{1.0 / (DT * NavEnvConfig.frame_skip):.0f} RL steps/s)",
    )
    parser.add_argument("--fast", action="store_true", help="No frame sleep")
    args = parser.parse_args()

    step_sleep_s = resolve_gui_step_sleep_s(
        gui=args.gui,
        sleep=args.sleep,
        realtime=args.realtime,
        fast=args.fast,
        frame_skip=NavEnvConfig.frame_skip,
    )

    run_manual_mixer(
        gui=args.gui,
        no_wind=args.no_wind,
        step_sleep_s=step_sleep_s,
        mix_strength=args.mix_strength,
    )


if __name__ == "__main__":
    main()
