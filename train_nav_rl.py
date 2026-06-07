"""
Train PPO to fly from spawn to a random goal using ``QuadNavGymEnv``.

Examples:
  python train_nav_rl.py
  python train_nav_rl.py --timesteps 500000 --n-envs 4
  python train_nav_rl.py --resume runs/nav_ppo/best/best_model.zip --timesteps 400000
  python train_nav_rl.py --eval --model runs/nav_ppo/best/best_model.zip --gui
"""

from __future__ import annotations

import project_env  # noqa: F401 — .env overrides shell before any config load

import argparse
from pathlib import Path

import numpy as np

from env_config import (
    build_nav_env_config,
    build_ppo_learning_rate,
    describe_nav_wind_settings,
    describe_ppo_checkpoint_settings,
    describe_ppo_early_stop_settings,
    describe_ppo_learning_rate,
    load_motor_thrust_settings,
    load_nav_action_mode,
    load_ppo_checkpoint_settings,
    load_ppo_early_stop_settings,
    load_ppo_eval_settings,
    load_ppo_lr_settings,
    load_ppo_resume_settings,
    describe_ppo_resume_settings,
    resolve_model_zip_path,
)
from ppo_callbacks import StopTrainingOnEvalPatience
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


def _create_ppo(
    vec_env,
    *,
    seed: int,
    learning_rate,
    n_steps: int,
    batch_size: int,
    gamma: float,
    ent_coef: float,
    tensorboard_log: str | None,
):
    from stable_baselines3 import PPO

    return PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        seed=seed,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        ent_coef=ent_coef,
        clip_range=0.2,
        tensorboard_log=tensorboard_log,
    )


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
    nav_cfg = build_nav_env_config(
        no_wind=args.no_wind,
        action_mode=load_nav_action_mode(action_mode=args.action_mode),
        motor_thrust_min=args.motor_thrust_min,
        motor_thrust_max=args.motor_thrust_max,
    )
    print(f"Action mode: {nav_cfg.action_mode}")
    print(f"Wind: {describe_nav_wind_settings(nav_cfg, cli_no_wind=args.no_wind)}")

    vec_env = build_vec_env(
        args.n_envs,
        args.seed,
        no_wind=args.no_wind,
        action_mode=load_nav_action_mode(action_mode=args.action_mode),
        motor_thrust_min=args.motor_thrust_min,
        motor_thrust_max=args.motor_thrust_max,
    )
    eval_env = build_vec_env(
        1,
        args.seed + 1,
        no_wind=args.no_wind,
        action_mode=load_nav_action_mode(action_mode=args.action_mode),
        motor_thrust_min=args.motor_thrust_min,
        motor_thrust_max=args.motor_thrust_max,
    )

    lr_settings = load_ppo_lr_settings(
        lr=args.lr,
        lr_final=args.lr_final,
        schedule=args.lr_schedule,
    )
    learning_rate = build_ppo_learning_rate(lr_settings)
    print(f"Learning rate schedule: {describe_ppo_learning_rate(lr_settings)}")

    ckpt_settings = load_ppo_checkpoint_settings(
        save_ckpt=args.save_ckpt,
        ckpt_freq=args.save_freq,
        save_best=args.save_best,
        save_final=args.save_final,
    )
    print(f"Model saves: {describe_ppo_checkpoint_settings(ckpt_settings)}")

    eval_settings = load_ppo_eval_settings(
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
    )
    early_stop_settings = load_ppo_early_stop_settings(
        early_stop=args.early_stop,
        patience=args.early_stop_patience,
        min_delta=args.early_stop_min_delta,
        min_evals=args.early_stop_min_evals,
    )
    print(
        f"Eval: every {eval_settings.eval_freq:,} timesteps, "
        f"{eval_settings.n_eval_episodes} episodes"
    )
    print(f"Early stopping: {describe_ppo_early_stop_settings(early_stop_settings)}")

    resume_settings = load_ppo_resume_settings(
        resume=args.resume,
        load_optimizer=args.load_optimizer,
    )
    print(f"Resume: {describe_ppo_resume_settings(resume_settings)}")

    tb_log = str(out_dir / "tb") if args.tensorboard else None

    if resume_settings.model_path is not None:
        resume_path = resolve_model_zip_path(resume_settings.model_path)
        checkpoint = PPO.load(str(resume_path), env=vec_env, print_system_info=False)
        prev_timesteps = checkpoint.num_timesteps
        if resume_settings.load_optimizer:
            model = checkpoint
            model.learning_rate = learning_rate
            if args.tensorboard and model.tensorboard_log is None:
                model.tensorboard_log = tb_log
            print(
                f"  loaded weights + optimizer from checkpoint "
                f"(was {prev_timesteps:,} timesteps); training counter reset"
            )
        else:
            model = _create_ppo(
                vec_env,
                seed=args.seed,
                learning_rate=learning_rate,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                gamma=args.gamma,
                ent_coef=args.ent_coef,
                tensorboard_log=tb_log,
            )
            model.policy.load_state_dict(checkpoint.policy.state_dict())
            print(
                f"  loaded policy weights only from checkpoint "
                f"(was {prev_timesteps:,} timesteps); fresh optimizer; counter reset"
            )
    else:
        model = _create_ppo(
            vec_env,
            seed=args.seed,
            learning_rate=learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            gamma=args.gamma,
            ent_coef=args.ent_coef,
            tensorboard_log=tb_log,
        )

    callbacks = []
    if ckpt_settings.save_ckpt:
        callbacks.append(
            CheckpointCallback(
                save_freq=max(ckpt_settings.ckpt_freq // args.n_envs, 1),
                save_path=str(out_dir / "ckpt"),
            )
        )
    early_stop_cb = None
    if early_stop_settings.enabled:
        early_stop_cb = StopTrainingOnEvalPatience(
            patience=early_stop_settings.patience,
            min_delta=early_stop_settings.min_delta,
            min_evals=early_stop_settings.min_evals,
            verbose=1,
        )

    callbacks.append(
        EvalCallback(
            eval_env,
            best_model_save_path=str(out_dir / "best") if ckpt_settings.save_best else None,
            log_path=str(out_dir / "eval"),
            eval_freq=max(eval_settings.eval_freq // args.n_envs, 1),
            n_eval_episodes=eval_settings.n_eval_episodes,
            deterministic=True,
            callback_after_eval=early_stop_cb,
        )
    )

    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        progress_bar=args.progress_bar,
        reset_num_timesteps=True,
    )
    print(f"Training finished at {model.num_timesteps:,} total timesteps")
    if ckpt_settings.save_final:
        final_path = out_dir / "final_model"
        model.save(final_path)
        print(f"Saved final model to {final_path}.zip")
    else:
        print("Skipping final model save (PPO_SAVE_FINAL=0)")
    vec_env.close()
    eval_env.close()
    return out_dir


def evaluate(args: argparse.Namespace) -> None:
    from stable_baselines3 import PPO

    model_path = str(resolve_model_zip_path(args.model))

    cfg = build_nav_env_config(
        gui=args.gui,
        step_sleep_s=args.sleep,
        no_wind=args.no_wind,
        action_mode=load_nav_action_mode(action_mode=args.action_mode),
        motor_thrust_min=args.motor_thrust_min,
        motor_thrust_max=args.motor_thrust_max,
        gui_realtime=args.realtime,
        gui_fast=args.fast,
    )
    print(f"Action mode: {cfg.action_mode}")
    print(f"Wind: {describe_nav_wind_settings(cfg, cli_no_wind=args.no_wind)}")
    if args.no_viz:
        cfg.show_wind_visualization = False
        cfg.show_thrust_visualization = False
    env = QuadNavEnv(cfg)
    model = PPO.load(model_path)

    if cfg.unlimited_episode:
        print("Episode horizon: unlimited (GUI_UNLIMITED_EPISODE — until success or crash)")

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
        elif reason == "unsafe_attitude":
            crashes += 1
        tilt = info.get("tilt_deg")
        tilt_s = f"  tilt={tilt:.1f}°" if tilt is not None else ""
        print(
            f"Episode {ep + 1}: {reason}  reward={total_r:+.1f}{tilt_s}  "
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
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="PATH",
        help="Load checkpoint before training; default PPO_RESUME_MODEL in .env",
    )
    parser.add_argument(
        "--load-optimizer",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="With --resume: also restore Adam state; default PPO_RESUME_LOAD_OPTIMIZER in .env",
    )
    parser.add_argument("--output", type=str, default="runs/nav_ppo")
    parser.add_argument("--timesteps", type=int, default=400_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Initial PPO learning rate; default from PPO_LEARNING_RATE in .env (3e-4)",
    )
    parser.add_argument(
        "--lr-schedule",
        choices=("constant", "linear", "cosine"),
        default=None,
        help="LR schedule; default from PPO_LR_SCHEDULE in .env (linear)",
    )
    parser.add_argument(
        "--lr-final",
        type=float,
        default=None,
        help="Final LR when schedule is linear/cosine; default PPO_LR_FINAL in .env (1e-5)",
    )
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument(
        "--save-freq",
        type=int,
        default=None,
        help="Periodic ckpt interval in timesteps; default PPO_CKPT_FREQ in .env (50000)",
    )
    parser.add_argument(
        "--save-ckpt",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Write runs/.../ckpt/ during training; default PPO_SAVE_CKPT in .env",
    )
    parser.add_argument(
        "--save-best",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Save best eval model to runs/.../best/; default PPO_SAVE_BEST in .env",
    )
    parser.add_argument(
        "--save-final",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Save final_model.zip at end; default PPO_SAVE_FINAL in .env",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=None,
        help="Eval every N timesteps; default PPO_EVAL_FREQ in .env (20000)",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=None,
        help="Episodes per eval round; default PPO_EVAL_EPISODES in .env (5)",
    )
    parser.add_argument(
        "--early-stop",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Stop when eval reward plateaus; default PPO_EARLY_STOP in .env",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=None,
        help="Eval rounds without improvement before stop; default PPO_EARLY_STOP_PATIENCE",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=None,
        help="Min eval reward gain to count as improvement; default PPO_EARLY_STOP_MIN_DELTA",
    )
    parser.add_argument(
        "--early-stop-min-evals",
        type=int,
        default=None,
        help="Min eval rounds before early stop can trigger; default PPO_EARLY_STOP_MIN_EVALS",
    )
    parser.add_argument("--episodes", type=int, default=5, help="Eval episode count (manual --eval mode)")
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--progress-bar", action="store_true", help="Requires tqdm and rich")
    parser.add_argument(
        "--no-wind",
        action="store_true",
        help="Disable wind for this run (overrides WIND_ENABLED in .env)",
    )
    parser.add_argument(
        "--action-mode",
        choices=("motors", "mixer"),
        default=None,
        help="motors = 4 thrusts; mixer = thrust+roll/pitch/yaw mix; default NAV_ACTION_MODE in .env",
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
