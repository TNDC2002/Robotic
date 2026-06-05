"""
Train PPO to fly from spawn to a random goal using ``QuadNavGymEnv``.

Examples:
  python train_nav_rl.py
  python train_nav_rl.py --timesteps 500000 --n-envs 4
  python train_nav_rl.py --eval --model runs/nav_ppo/best_model.zip --gui
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from env_config import build_nav_env_config, load_motor_thrust_settings
from quad_nav_env import QuadNavEnv, QuadNavGymEnv, make_nav_env


def _require_gym():
    if QuadNavGymEnv is None:
        raise SystemExit(
            "Install RL dependencies: pip install gymnasium stable-baselines3 shimmy"
        )


def build_vec_env(
    n_envs: int,
    seed: int,
    *,
    no_wind: bool,
    action_mode: str,
    motor_thrust_min: float | None = None,
    motor_thrust_max: float | None = None,
):
    _require_gym()
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    def _factory():
        cfg = build_nav_env_config(
            gui=False,
            step_sleep_s=0.0,
            no_wind=no_wind,
            action_mode=action_mode,  # type: ignore[arg-type]
            motor_thrust_min=motor_thrust_min,
            motor_thrust_max=motor_thrust_max,
        )
        return Monitor(QuadNavGymEnv(cfg))

    # DummyVecEnv: one PyBullet client per env, reliable on Windows.
    return make_vec_env(_factory, n_envs=n_envs, seed=seed, vec_env_cls=DummyVecEnv)


def train(args: argparse.Namespace) -> Path:
    _require_gym()
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    thrust = load_motor_thrust_settings(
        min_n=args.motor_thrust_min,
        max_n=args.motor_thrust_max,
    )
    print(
        f"Motor thrust map: action [-1,1] -> [{thrust.min_n}, {thrust.max_n}] N per rotor "
        f"(linear: thrust = {thrust.min_n} + ({thrust.max_n}-{thrust.min_n})*(action+1)/2)"
    )

    vec_env = build_vec_env(
        args.n_envs,
        args.seed,
        no_wind=args.no_wind,
        action_mode=args.action_mode,
        motor_thrust_min=args.motor_thrust_min,
        motor_thrust_max=args.motor_thrust_max,
    )
    eval_env = build_vec_env(
        1,
        args.seed + 1,
        no_wind=args.no_wind,
        action_mode=args.action_mode,
        motor_thrust_min=args.motor_thrust_min,
        motor_thrust_max=args.motor_thrust_max,
    )

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        seed=args.seed,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        ent_coef=args.ent_coef,
        clip_range=0.2,
        tensorboard_log=str(out_dir / "tb") if args.tensorboard else None,
    )

    callbacks = [
        CheckpointCallback(save_freq=max(args.save_freq // args.n_envs, 1), save_path=str(out_dir / "ckpt")),
    ]
    callbacks.append(
        EvalCallback(
            eval_env,
            best_model_save_path=str(out_dir / "best"),
            log_path=str(out_dir / "eval"),
            eval_freq=max(args.eval_freq // args.n_envs, 1),
            n_eval_episodes=args.eval_episodes,
            deterministic=True,
        )
    )

    model.learn(total_timesteps=args.timesteps, callback=callbacks, progress_bar=args.progress_bar)
    final_path = out_dir / "final_model"
    model.save(final_path)
    print(f"Saved final model to {final_path}.zip")
    vec_env.close()
    eval_env.close()
    return out_dir


def evaluate(args: argparse.Namespace) -> None:
    from stable_baselines3 import PPO

    model_path = args.model
    if not model_path.endswith(".zip"):
        model_path += ".zip"
    if not os.path.isfile(model_path):
        raise FileNotFoundError(model_path)

    cfg = build_nav_env_config(
        gui=args.gui,
        step_sleep_s=args.sleep,
        motor_thrust_min=args.motor_thrust_min,
        motor_thrust_max=args.motor_thrust_max,
        gui_realtime=args.realtime,
        gui_fast=args.fast,
    )
    if args.no_viz:
        cfg.show_wind_visualization = False
        cfg.show_thrust_visualization = False
    env = QuadNavEnv(cfg)
    model = PPO.load(model_path)

    if args.gui:
        step_sleep_s = cfg.step_sleep_s
        if step_sleep_s > 0.0:
            print(
                f"Playback: {step_sleep_s * 1000:.1f} ms per RL step "
                f"(~{1.0 / step_sleep_s:.0f} agent steps/s view)"
            )
        else:
            print("Playback: max speed (use --sleep 0.01 or --realtime to slow down)")

    successes = 0
    crashes = 0
    for ep in range(args.episodes):
        obs = env.reset(seed=args.seed + ep if args.seed is not None else None)
        done = False
        total_r = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_r += reward
        reason = info.get("terminal_reason")
        if reason == "success":
            successes += 1
        elif reason == "crash":
            crashes += 1
        print(
            f"Episode {ep + 1}: {reason}  reward={total_r:+.1f}  "
            f"dist={info.get('distance_to_goal', 0):.3f} m  goal={info.get('goal')}"
        )
    env.close()
    print(f"Summary: {successes}/{args.episodes} success, {crashes}/{args.episodes} crash")


def smoke_test() -> None:
    """Quick random-action rollout (no SB3)."""
    env = make_nav_env()
    obs = env.reset(seed=0)
    total = 0.0
    for _ in range(200):
        action = np.random.uniform(-1, 1, size=4).astype(np.float32)
        obs, reward, done, info = env.step(action)
        total += reward
        if done:
            print("smoke:", info["terminal_reason"], "reward", total)
            break
    env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train / eval quad navigation PPO")
    parser.add_argument("--eval", action="store_true", help="Run evaluation instead of training")
    parser.add_argument("--smoke", action="store_true", help="Random-action env smoke test")
    parser.add_argument("--gui", action="store_true", help="GUI during evaluation")
    parser.add_argument("--model", type=str, default="runs/nav_ppo/best/best_model.zip")
    parser.add_argument("--output", type=str, default="runs/nav_ppo")
    parser.add_argument("--timesteps", type=int, default=400_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--save-freq", type=int, default=50_000)
    parser.add_argument("--eval-freq", type=int, default=20_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=5, help="Eval episode count")
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--progress-bar", action="store_true", help="Requires tqdm and rich")
    parser.add_argument("--no-wind", action="store_true", help="Disable per-episode random wind")
    parser.add_argument(
        "--action-mode",
        choices=("motors", "mixer"),
        default="motors",
        help="motors = 4 thrusts; mixer = thrust+attitude mix (easier)",
    )
    parser.add_argument(
        "--motor-thrust-min",
        type=float,
        default=None,
        help="Per-motor thrust at action -1 (N); overrides MOTOR_THRUST_MIN_N in .env",
    )
    parser.add_argument(
        "--motor-thrust-max",
        type=float,
        default=None,
        help="Per-motor thrust at action +1 (N); overrides MOTOR_THRUST_MAX_N in .env",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=None,
        metavar="SEC",
        help="GUI eval: pause once per RL step (default 0.01 s when --gui)",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="GUI eval: match sim time (~60 agent steps/s with frame_skip=4)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="GUI eval: no frame sleep (max speed)",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="GUI eval: disable force/wind debug arrows (smoother FPS)",
    )
    args = parser.parse_args()

    if args.smoke:
        smoke_test()
    elif args.eval:
        evaluate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
