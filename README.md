# Robotic Quad Navigation RL

PyBullet quadrotor sim with PPO navigation training (`train_nav_rl.py`), hover demo (`rl_interact_demo.py`), and configurable wind / thrust via `.env`.

## Setup

```bash
cd Robotic
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # edit thrust, wind, GUI speed
```

## Configuration (`.env`)

Settings load from `.env` automatically (`env_config.py`). **CLI flags override `.env`.**

### Motor thrust (navigation RL)

Linear map from policy action `[-1, 1]` to per-motor thrust (N):

```text
thrust_N = MIN + (MAX - MIN) * (action + 1) / 2
```

| Variable | Example | Meaning |
|----------|---------|---------|
| `MOTOR_THRUST_MIN_N` | `0` | Thrust at action `-1` |
| `MOTOR_THRUST_MAX_N` | `4` | Thrust at action `+1` |

Example: `MIN=0`, `MAX=4` → `-1→0 N`, `-0.5→1 N`, `0→2 N`, `0.5→3 N`, `1→4 N`.

### GUI playback

| Variable | Default | Meaning |
|----------|---------|---------|
| `GUI_STEP_SLEEP_S` | `0.01` | Pause per RL step (nav eval) or physics step (hover demo) |
| `GUI_REALTIME` | `0` | `1` = match sim clock |
| `GUI_FAST` | `0` | `1` = no sleep (max speed) |

### Fixed wind (demo / `--no-wind` training)

Used by `rl_interact_demo.py` and when `WIND_RANDOMIZATION=0`.

| Variable | Meaning |
|----------|---------|
| `WIND_ENABLED` | `1` = wind on |
| `WIND_VX`, `WIND_VY`, `WIND_VZ` | Mean wind velocity (m/s) |
| `WIND_DRAG`, `WIND_QUAD_DRAG` | Drag coefficients |
| `WIND_GUST` | Sinusoidal gust amplitude (m/s) |
| `WIND_TURBULENCE_SCALE` | Multiplier on base turbulence |
| `WIND_TURBULENCE_BASE_X/Y/Z` | OU turbulence std (m/s) |
| `WIND_FORCE_NOISE`, `WIND_CORNER_NOISE`, `WIND_TORQUE_NOISE` | Extra noise (N or N·m) |
| `WIND_FORCE_NOISE_Z_SCALE`, `WIND_CORNER_NOISE_Z_SCALE` | Softer vertical noise |
| `WIND_VERTICAL_GUST_COUPLING` | Vertical fraction of gust |

See `.env.example` for the full list.

### Random wind (navigation training)

When `WIND_RANDOMIZATION=1` (default for training), each episode samples wind from `*_MIN` / `*_MAX` ranges (`WIND_SPEED_MIN`, `WIND_DRAG_MIN`, etc.).

Training with fixed wind instead:

```bash
# .env: WIND_RANDOMIZATION=0 and set WIND_VX, WIND_VY, ...
python train_nav_rl.py --no-wind   # or configure fixed WIND_* in .env
```

## Wind-tunnel / hover-balance mode

Hold thrust fixed at **m·g / 4 per motor** (total = weight) so the drone floats while wind pushes it — useful to inspect force arrows and wind settings without a trained policy.

```bash
# CLI
python rl_interact_demo.py --gui --hover-balance --seconds 60

# Or .env: HOVER_BALANCE_MODE=1
python rl_interact_demo.py --gui --seconds 60
```

Tune wind in `.env` (`WIND_VX`, `WIND_DRAG`, turbulence, noise) and re-run. Debug arrows show drag vs thrust vs weight in **Newtons** (same scale).

## Training

```bash
python train_nav_rl.py --timesteps 400000 --n-envs 8 --n-steps 1024 --batch-size 512
```

Disable random wind:

```bash
python train_nav_rl.py --no-wind
```

### Outputs

- `runs/nav_ppo/ckpt/` — periodic checkpoints
- `runs/nav_ppo/best/best_model.zip` — best eval model
- `runs/nav_ppo/final_model.zip` — model at end of training

### Resume training

No `--resume` flag; load a checkpoint with SB3 and `reset_num_timesteps=False` (see example in git history or SB3 docs).

## Evaluation

```bash
python train_nav_rl.py --eval --gui --model runs/nav_ppo/best/best_model.zip
python train_nav_rl.py --eval --gui --sleep 0.02 --no-viz --model runs/nav_ppo/best/best_model.zip
```

Uses `.env` for thrust range, wind randomization, and `GUI_STEP_SLEEP_S`.

## Hover demo (no RL)

```bash
python rl_interact_demo.py --gui
python rl_interact_demo.py --gui --hover-balance --seconds 120
python rl_interact_demo.py --gui --wind 1.5 0 0    # CLI overrides WIND_VX/Y/Z
```

## SLURM

```bash
./submit_train_nav_rl.sh
```

Logs: `logs/slurm-<jobid>.out`, `logs/job_<jobid>.log`.

## Project layout

| File | Role |
|------|------|
| `quad_hover_env.py` | Hover physics, wind, force visualization |
| `quad_nav_env.py` | Spawn → goal navigation task |
| `train_nav_rl.py` | PPO training / eval |
| `rl_interact_demo.py` | Readable hover / wind-tunnel demo |
| `env_config.py` | `.env` loader |
| `.env` | Local config (gitignored) |
