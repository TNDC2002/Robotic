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

### Motor thrust (both action modes)

Per-rotor thrust limits (N) from `.env` — used by **`motors`** and **`mixer`**:

```text
thrust_N = MIN + (MAX - MIN) * (action + 1) / 2   # motors mode only (linear map)
each motor clipped to [MIN, MAX]                   # both modes
```

| Variable | Example | Meaning |
|----------|---------|---------|
| `MOTOR_THRUST_MIN_N` | `0` | Minimum thrust per motor (N) |
| `MOTOR_THRUST_MAX_N` | `3` | **Maximum thrust per motor (N)** — caps mixer roll/pitch/yaw mix too |

In **mixer** mode, `a0` still adjusts total thrust around m·g, but no motor exceeds `MOTOR_THRUST_MAX_N`.

### Action mode (`NAV_ACTION_MODE`)

The policy **always** outputs **4** numbers in `[-1, 1]` (Gym `Box(4,)`). What they mean depends on the mode:

| Mode | `.env` | Action vector `[a0, a1, a2, a3]` |
|------|--------|-------------------------------------|
| **`motors`** (default) | `NAV_ACTION_MODE=motors` | **Four motor thrusts** — linear map via `MOTOR_THRUST_MIN_N` / `MOTOR_THRUST_MAX_N` |
| **`mixer`** | `NAV_ACTION_MODE=mixer` | **`[thrust_delta, roll_mix, pitch_mix, yaw_mix]`** — total thrust ≈ m·g when `a0=0`; roll/pitch mix tilt; yaw differential |

Mixer decode (`QuadNavEnv._decode_action_mixer`): `a0=0, a1=a2=a3=0` → hover float (~m·g/4 per motor). Not a 2-element action — still 4D, but only thrust/roll/pitch/yaw semantics instead of four independent thrusts.

```env
NAV_ACTION_MODE=motors   # or mixer
```

CLI override: `--action-mode mixer`. Must match the mode used when the checkpoint was trained.

Try mixer manually: `python manual_mixer_demo.py --gui`

### GUI playback

| Variable | Default | Meaning |
|----------|---------|---------|
| `GUI_STEP_SLEEP_S` | `0.01` | Pause per RL step (nav eval) or physics step (hover demo) |
| `GUI_REALTIME` | `0` | `1` = match sim clock |
| `GUI_FAST` | `0` | `1` = no sleep (max speed) |

### Wind (nav train/eval + hover demo)

Each episode draws wind uniformly from `*_MIN` / `*_MAX` ranges in `.env` (horizontal speed + random direction). Same sampling for **training**, **`train_nav_rl.py --eval`**, and **`rl_interact_demo.py`** (QuadNavEnv).

| Variable | Meaning |
|----------|---------|
| `WIND_ENABLED` | `0` = no wind (or use `--no-wind`) |
| `WIND_INCLUDE_IN_OBS` | Append effective wind to RL observations |
| `WIND_SEED` | Optional; reproducible episode sampling |
| `WIND_SPEED_MIN/MAX` | Horizontal wind speed (m/s) |
| `WIND_DRAG_MIN/MAX` | Linear drag coefficient |
| `WIND_QUAD_DRAG_MIN/MAX` | Quadratic drag |
| `WIND_TURBULENCE_SCALE_MIN/MAX` | Turbulence multiplier |
| `WIND_FORCE_NOISE_MIN/MAX` | Extra drag force noise (N) |
| `WIND_CORNER_NOISE_MIN/MAX` | Per-motor corner noise (N) |
| `WIND_TORQUE_NOISE_MIN/MAX` | Torque noise (N·m) |
| `WIND_GUST_MIN/MAX` | Sinusoidal gust amplitude (m/s) |

```bash
python train_nav_rl.py --no-wind          # disable for one run
python rl_interact_demo.py --gui --no-wind
```

See `.env.example` for defaults.

**Config priority:** explicit CLI flags (when you pass them) → **`.env`** (overrides shell/conda env) → code defaults. Restart training after editing `.env`.

### Navigation reward weights

Tune PPO shaping via `REWARD_*` (bonuses) and `PENALTY_*` (costs) in `.env` (see [REWARD.md](REWARD.md) for formulas):

```env
REWARD_W_PROGRESS=8.0
REWARD_SUCCESS_BONUS=120.0
PENALTY_CRASH=80.0
```

## Wind-tunnel / hover-balance preview

Uses **`QuadNavEnv`** — the same environment and per-episode wind sampling as PPO **train** and **`--eval`**. Each reset draws wind from `WIND_*_MIN/MAX` in `.env`.

```bash
python rl_interact_demo.py --gui --hover-balance --seconds 60
python rl_interact_demo.py --gui --no-wind   # disable wind
```

Fixed **m·g/4** per motor (`--hover-balance`, default on in this script) so you can watch force arrows without a trained policy. The printed **sampled episode wind** line matches what navigation training sees at reset.

Tune ranges in `.env` (`WIND_SPEED_MIN/MAX`, drag, turbulence, noise).

## Manual mixer control (keyboard)

PPO can use **`mixer`** action mode instead of four raw motor thrusts (`train_nav_rl.py --action-mode mixer`). The policy outputs total-thrust delta + roll/pitch/yaw mix; see `QuadNavEnv._decode_action_mixer()`.

Try it yourself with the keyboard demo (same mixer decode as training):

```bash
python manual_mixer_demo.py --gui
python manual_mixer_demo.py --gui --no-wind
```

| Key | Effect |
|-----|--------|
| **W / S** | Pitch command (forward / back) |
| **A / D** | Roll command (left / right) |
| **(release all)** | `[0,0,0,0]` → thrust ≈ m·g, no mix → **hover float** (~m·g/4 per motor) |
| **ESC** | Quit |

Click the PyBullet window so it receives key events.

## Training

```bash
python train_nav_rl.py --timesteps 400000 --n-envs 8 --n-steps 1024 --batch-size 512
python train_nav_rl.py --action-mode mixer   # roll/pitch mix instead of 4 motor thrusts
```

Learning rate schedule (`.env` or CLI `--lr`, `--lr-schedule`, `--lr-final`):

```env
PPO_LEARNING_RATE=0.0003
PPO_LR_FINAL=0.00001
PPO_LR_SCHEDULE=linear   # constant | linear | cosine
```

Disable decay: `PPO_LR_SCHEDULE=constant`, or set `PPO_LR_FINAL` equal to `PPO_LEARNING_RATE`.

Model saves (`.env` or CLI `--save-ckpt`, `--no-save-ckpt`, `--save-freq`, etc.):

```env
PPO_SAVE_CKPT=1          # periodic ckpt/ (0 = best + final only)
PPO_CKPT_FREQ=50000      # timesteps between ckpt saves
PPO_SAVE_BEST=1          # best/ from eval callback
PPO_SAVE_FINAL=1         # final_model.zip at end
```

Eval + early stopping (optional; uses the same eval env as `best/` saves):

```env
PPO_EVAL_FREQ=20000           # timesteps between eval rounds
PPO_EVAL_EPISODES=5           # episodes averaged per eval
PPO_EARLY_STOP=1              # 1 = stop when eval reward plateaus
PPO_EARLY_STOP_PATIENCE=5     # eval rounds without improvement
PPO_EARLY_STOP_MIN_DELTA=0.0  # minimum reward gain to reset patience
PPO_EARLY_STOP_MIN_EVALS=3    # warmup evals before stop can trigger
```

RL does not require early stopping, but it saves time when the policy has plateaued. Keep `PPO_SAVE_BEST=1` so the best checkpoint is kept even if training stops early.

Disable wind for one run:

```bash
python train_nav_rl.py --no-wind
```

### Resume training

Load a checkpoint before training. The **timestep counter always resets**; `--timesteps` is the full budget for this run.

```env
PPO_RESUME_MODEL=runs/nav_ppo/best/best_model.zip
PPO_RESUME_LOAD_OPTIMIZER=0
```

| `PPO_RESUME_LOAD_OPTIMIZER` | Behavior |
|-----------------------------|----------|
| `0` (default) | **Policy weights only** — fresh optimizer, uses current `.env` / CLI hyperparams (`n-steps`, `batch-size`, LR, …) |
| `1` | **Full checkpoint** — weights + Adam state (keeps saved `n_steps` / `batch_size` from the zip) |

```bash
python train_nav_rl.py --timesteps 4000000
python train_nav_rl.py --resume runs/nav_ppo/final_model.zip --no-load-optimizer
```

Leave `PPO_RESUME_MODEL` empty to train from scratch. Observation size must match the saved model.

### Outputs

- `runs/nav_ppo/ckpt/` — periodic checkpoints (when `PPO_SAVE_CKPT=1`)
- `runs/nav_ppo/best/best_model.zip` — best eval model (when `PPO_SAVE_BEST=1`)
- `runs/nav_ppo/final_model.zip` — model at end of training (when `PPO_SAVE_FINAL=1`)

Use `--resume` or `PPO_RESUME_MODEL` in `.env` to continue from `best/` or `final_model.zip` (see **Resume training** above).

## Evaluation

```bash
python train_nav_rl.py --eval --gui --model runs/nav_ppo/best/best_model.zip
python train_nav_rl.py --eval --gui --sleep 0.02 --no-viz --model runs/nav_ppo/best/best_model.zip
```

Uses `.env` for thrust range, wind ranges, and `GUI_STEP_SLEEP_S`.

Unlimited GUI episodes (no ~60 s timeout): set `GUI_UNLIMITED_EPISODE=1` — episode ends only on **success** or **crash**.

## Hover demo (no RL)

```bash
python rl_interact_demo.py --gui
python rl_interact_demo.py --gui --hover-balance --seconds 120
python rl_interact_demo.py --gui --no-wind
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
| `REWARD.md` | Reward function reference (navigation + hover) |
| `.env` | Local config (gitignored) |
