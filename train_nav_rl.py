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

from quad_nav_env import NavEnvConfig, NavWindRandomizationConfig, QuadNavEnv, QuadNavGymEnv, make_nav_env


def _require_gym():
    if QuadNavGymEnv is None:
        raise SystemExit(
            "Install RL dependencies: pip install gymnasium stable-baselines3 shimmy"
        )


def build_vec_env(n_envs: int, seed: int, *, no_wind: bool, action_mode: str):
    _require_gym()
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    def _factory():
        wr = NavWindRandomizationConfig(enabled=not no_wind)
        cfg = NavEnvConfig(
            gui=False,
            step_sleep_s=0.0,
            wind_randomization=wr,
            action_mode=action_mode,  # type: ignore[arg-type]
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

    vec_env = build_vec_env(args.n_envs, args.seed, no_wind=args.no_wind, action_mode=args.action_mode)
    eval_env = build_vec_env(1, args.seed + 1, no_wind=args.no_wind, action_mode=args.action_mode)

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

    cfg = NavEnvConfig(gui=args.gui, step_sleep_s=0.01 if args.gui else 0.0)
    env = QuadNavEnv(cfg)
    model = PPO.load(model_path)

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
    args = parser.parse_args()

    if args.smoke:
        smoke_test()
    elif args.eval:
        evaluate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
