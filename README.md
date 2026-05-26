## Robotic Quad Navigation RL

This repo trains a PPO agent to fly a quadrotor from a spawn point to a random goal using `QuadNavGymEnv` (`train_nav_rl.py`).

### Prerequisites

Create/activate the project venv (already present in this repo as `.venv`):

```bash
cd /home/chuongtnd/git-repo/Robotic
source .venv/bin/activate
pip install -r requirements.txt
```

### Training (local)

Run training with default settings:

```bash
python train_nav_rl.py
```

Example (matches your request):

```bash
python train_nav_rl.py --timesteps 4000000 --n-envs 16
```

### Outputs / checkpoints

By default training writes to:

- `runs/nav_ppo/`
- checkpoints: `runs/nav_ppo/ckpt/rl_model_<N>_steps.zip`
- best model: `runs/nav_ppo/best/best_model.zip`
- final model: `runs/nav_ppo/final_model.zip`

### Continue training (resume from an existing checkpoint)

There is **no `--resume` flag** in `train_nav_rl.py`. To continue training, you need to load the checkpoint with Stable-Baselines3, then call `learn(..., reset_num_timesteps=False)` so the timestep counter is treated as cumulative.

Example: resume to `4_000_000` total timesteps from the latest checkpoint:

```bash
python -c "
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from train_nav_rl import build_vec_env

out_dir = Path('runs/nav_ppo')
ckpt = 'runs/nav_ppo/ckpt/rl_model_100000_steps.zip'  # <-- change if you use another checkpoint

n_envs = 16
seed = 0
no_wind = False
action_mode = 'motors'

vec_env = build_vec_env(n_envs, seed, no_wind=no_wind, action_mode=action_mode)
eval_env = build_vec_env(1, seed + 1, no_wind=no_wind, action_mode=action_mode)

model = PPO.load(ckpt, env=vec_env)

callbacks = [
  CheckpointCallback(
    save_freq=max(50_000 // n_envs, 1),
    save_path=str(out_dir / 'ckpt')
  ),
  EvalCallback(
    eval_env,
    best_model_save_path=str(out_dir / 'best'),
    log_path=str(out_dir / 'eval'),
    eval_freq=max(20_000 // n_envs, 1),
    n_eval_episodes=5,
    deterministic=True
  )
]

model.learn(
  total_timesteps=4_000_000,
  callback=callbacks,
  reset_num_timesteps=False,
  progress_bar=True
)

model.save(out_dir / 'final_model')
vec_env.close()
eval_env.close()
"
```

Tip: make sure you use the same `--n-envs`, `--seed`, and `--action-mode` (and `--no-wind` / wind randomization) that match your earlier run as closely as possible.

### Evaluation

Run evaluation only (uses the default `runs/nav_ppo/best/best_model.zip`):

```bash
python train_nav_rl.py --eval --episodes 5
```

### SLURM submission

This repo includes two scripts in the project root:

- `submit_train_nav_rl.sh`: convenience wrapper that calls `sbatch` on the `.slurm` file
- `submit_train_nav_rl.slurm`: the actual SLURM batch script

Submit:

```bash
cd /home/chuongtnd/git-repo/Robotic
./submit_train_nav_rl.sh
```

The current `submit_train_nav_rl.slurm` settings in this repo are:

- `--partition=mig`
- `--cpus-per-task=8`
- `--mem=96G`
- `--time=0` (unlimited walltime on this cluster; override with `./submit_train_nav_rl.sh --time=HH:MM:SS`)
- training command uses `--n-envs 8`, `--n-steps 1024`, `--batch-size 512`

If you want `16` CPU cores and `--n-envs 16`, edit `submit_train_nav_rl.slurm` accordingly.

#### Debugging SLURM jobs

Always submit via `./submit_train_nav_rl.sh` (it creates `logs/` before `sbatch` and uses absolute log paths).

Log files:

- `logs/slurm-<jobid>.out` — SLURM stdout+stderr (same file)
- `logs/job_<jobid>.log` — backup copy inside the repo (tee)

If a job dies immediately, check scheduler state:

```bash
sacct -j <jobid> --format=JobID,State,ExitCode,Elapsed,MaxRSS,ReqMem,Reason
```

Common instant-fail causes: partition/account limits, missing `logs/` at submit time, or invalid `--time` syntax on clusters that do not accept `0` as unlimited.

