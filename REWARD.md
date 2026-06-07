# Reward functions

This project has two environments with different rewards:

| Environment | File | Used for |
|-------------|------|----------|
| **Navigation** | `quad_nav_env.py` | PPO training (`train_nav_rl.py`) — fly spawn → random goal |
| **Hover** | `quad_hover_env.py` | Baseline demo (`rl_interact_demo.py`) — hold a fixed pose |

The sections below focus on **navigation**, since that is what the RL agent optimizes.

---

## Navigation reward (PPO)

Implemented in `QuadNavEnv._compute_nav_reward()`. Each **physics substep** (240 Hz) adds a reward; one RL `step()` runs `frame_skip` substeps (default **4**), so the return from `env.step()` is the **sum of up to 4 substep rewards**.

### Total per substep

```text
R = progress + goal_dist + goal_alt + goal_alt_progress + alive + attitude + ang_vel + lin_vel + upright + action_rate + safe_attitude + unsafe_attitude + terminal
```

`terminal` is non-zero only on the substep where the episode ends.

### Term-by-term

| Term | Formula | Default weight | Sign | Purpose |
|------|---------|----------------|------|---------|
| **progress** | `w_progress × (d_prev − d_now)` | `8.0` | ± | Dense reward for moving **closer** to the goal. `d` is 3D Euclidean distance to goal. This is the main learning signal. |
| **goal_dist** | `−w_goal_dist × d_now` | `0.4` | − | Small shaping penalty proportional to **remaining 3D distance**. Keeps pressure on the agent even when progress per step is tiny. |
| **goal_alt** | `−w_goal_alt × |z − z_goal|` | `1.0` | − | Penalize vertical offset from the **goal altitude** (m). |
| **goal_alt_progress** | `w_goal_alt_progress × (e_prev − e_now)` | `4.0` | ± | Reward **closing** vertical gap; `e = |z − z_goal|`. Mirrors `progress` but altitude-only. |
| **alive** | `+w_alive` | `0.02` | + | Constant per-substep bonus for surviving. Encourages not crashing. |
| **attitude** | `−w_attitude × tilt_rad` | `1.2` | − | Penalize tilt from vertical; `tilt_rad = acos(uprightness)`. |
| **ang_vel** | `−w_ang_vel × ‖ω‖` | `0.06` | − | Penalize angular velocity (rad/s). Reduces spinning and wobble. |
| **lin_vel** | `−w_lin_vel × ‖v‖` | `0.04` | − | Penalize linear speed (m/s). Encourages smoother, less frantic flight. |
| **upright** | `+w_upright × max(0, uprightness)` | `0.15` | + | Bonus when body +Z aligns with world +Z (`uprightness` = cos(tilt)). |
| **action_rate** | `−w_action_rate × Σ(a − a_prev)²` | `0.02` | − | Penalize large changes in the 4D action vector. Smoother motor commands. |
| **safe_attitude** | `+w_safe_attitude` when `tilt_deg < SAFE_ATTITUDE_DEG` | `1.0` | + | Per-substep bonus for staying within the safe tilt envelope. |
| **unsafe_attitude** | `−w_unsafe_attitude` when `tilt_deg ≥ SAFE_ATTITUDE_DEG` | `5.0` | − | Extra per-substep penalty while tilted past the safe angle. Episode **continues** unless `NAV_END_ON_UNSAFE_ATTITUDE=1`. |
| **terminal** | see below | — | ± | One-shot bonus or penalty when the episode ends. |

**Variables**

- `d_now`, `d_prev`: distance from drone to goal (m).
- `z`, `z_goal`: drone and goal altitude (m).
- `e_now`, `e_prev`: `|z − z_goal|` (m).
- `roll`, `pitch`: body Euler angles (rad) — in observations only; rewards use **tilt from vertical**.
- `uprightness`: cos(tilt) = body +Z · world +Z; 1 = level, 0 = 90° bank, −1 = inverted.
- `tilt_deg`: `acos(uprightness)` in degrees — matches `SAFE_ATTITUDE_DEG`.
- `v`: linear velocity `(vx, vy, vz)` (m/s).
- `ω`: angular velocity `(wx, wy, wz)` (rad/s).
- `a`: current action in `[−1, 1]⁴`; `a_prev` is the action from the previous RL step.

### Terminal reward

Applied once when `done=True` on that substep:

| `terminal_reason` | Condition | Default value |
|-------------------|-----------|---------------|
| `success` | Within `goal_radius` (0.32 m) in XY and `goal_z_tolerance` (0.25 m) in Z | **+120** |
| `crash` | Altitude ≤ `crash_z` (0.12 m) — floor impact only | **−80** |
| `time_limit` | Physics step count ≥ `max_episode_steps` (14 400 ≈ 60 s) | **−15** |

Tilt past `SAFE_ATTITUDE_DEG` applies `unsafe_attitude` each substep until the drone recovers or the episode ends. While within the limit, `safe_attitude` applies instead. Set `NAV_END_ON_UNSAFE_ATTITUDE=1` to terminate immediately (terminal penalty `PENALTY_UNSAFE_ATTITUDE_END`).

**Important:** `SAFE_ATTITUDE_DEG` is **tilt from vertical** (body +Z vs world +Z), not separate roll/pitch caps.

### Episode flow and scaling

```text
One RL step (agent calls env.step(action)):
  repeat up to frame_skip (=4) physics substeps:
    apply forces, integrate, compute reward r_i
    if done: break
  return sum(r_i), done, info
```

Rough magnitude intuition (order of magnitude):

- **Progress**: moving 0.1 m closer in one substep → `8 × 0.1 = +0.8`.
- **Success**: `+120` once — dominates a good episode.
- **Crash**: `−80` once — strongly discourages failure.
- **Alive**: `0.02 × 4 substeps ≈ 0.08` per RL step if all substeps run.

Over a 60 s episode at 240 Hz with 4× frame skip, there are up to **3 600 RL steps** and **14 400** reward evaluations.

### Default weights (config)

All weights live on `NavEnvConfig` in `quad_nav_env.py` and can be set in **`.env`** (loaded by `env_config.load_nav_reward_settings()`). Use **`REWARD_*`** for bonuses and **`PENALTY_*`** for costs (legacy `REWARD_W_*` penalty names still work as fallbacks).

**Rewards (`REWARD_*`)**

| `.env` variable | Config field | Default |
|-----------------|--------------|---------|
| `REWARD_W_PROGRESS` | `w_progress` | `8.0` |
| `REWARD_W_GOAL_ALT_PROGRESS` | `w_goal_alt_progress` | `4.0` |
| `REWARD_W_ALIVE` | `w_alive` | `0.02` |
| `REWARD_W_UPRIGHT` | `w_upright` | `0.15` |
| `REWARD_W_SAFE_ATTITUDE` | `w_safe_attitude` | `1.0` |
| `REWARD_SUCCESS_BONUS` | `success_bonus` | `120.0` |

**Penalties (`PENALTY_*`)**

| `.env` variable | Config field | Default |
|-----------------|--------------|---------|
| `PENALTY_W_GOAL_DIST` | `w_goal_dist` | `0.4` |
| `PENALTY_W_GOAL_ALT` | `w_goal_alt` | `1.0` |
| `PENALTY_W_ATTITUDE` | `w_attitude` | `1.2` |
| `PENALTY_W_ANG_VEL` | `w_ang_vel` | `0.06` |
| `PENALTY_W_LIN_VEL` | `w_lin_vel` | `0.04` |
| `PENALTY_W_ACTION_RATE` | `w_action_rate` | `0.02` |
| `PENALTY_W_UNSAFE_ATTITUDE` | `w_unsafe_attitude` | `5.0` |
| `PENALTY_UNSAFE_ATTITUDE_END` | `penalty_unsafe_attitude_end` | `50.0` |
| `NAV_END_ON_UNSAFE_ATTITUDE` | `end_on_unsafe_attitude` | `0` |
| `PENALTY_CRASH` | `crash_penalty` | `80.0` |
| `PENALTY_TIME_LIMIT` | `time_limit_penalty` | `15.0` |

**Other**

| `.env` variable | Config field | Default |
|-----------------|--------------|---------|
| `SAFE_ATTITUDE_DEG` | `flip_angle_rad` (converted to rad) | `70` |

Training and eval via `build_nav_env_config()` pick these up automatically. Restart training after changing reward weights.

**Safe tilt:** when `tilt_deg = acos(uprightness)` is below `SAFE_ATTITUDE_DEG`, the `safe_attitude` bonus applies each substep. When tilt reaches the limit, `unsafe_attitude` replaces it. Optional: `NAV_END_ON_UNSAFE_ATTITUDE=1` ends the episode on excessive tilt.

### What the agent is encouraged to do (summary)

1. **Reach the goal** — large progress term + success bonus.
2. **Stay alive** — alive bonus and crash penalty.
3. **Fly smoothly** — attitude, velocity, action-rate penalties.
4. **Finish before timeout** — time limit penalty (weaker than crash).

Wind is **not** in the reward; the agent only sees wind in observations (when enabled). It must learn to compensate implicitly.

### Inspecting rewards during eval

`info` from `env.step()` includes a breakdown in `info["reward_parts"]` (last substep of the RL step):

```python
obs, reward, done, info = env.step(action)
print(info["reward_parts"])   # dict of term names → float
print(info["terminal_reason"])
```

---

## Hover reward (baseline)

Used by `QuadHoverEnv` and `rl_interact_demo.py`. The agent (or built-in PD autopilot) tries to **hold a fixed target pose**, not reach a moving goal.

### Total

```text
R = alive + position + attitude + velocity + upright_bonus + terminal_penalty
```

| Component | Formula | Default | Purpose |
|-----------|---------|---------|---------|
| **alive** | `+w_alive` | `0.05` | Survive each frame |
| **position** | `−w_pos_xy × err_xy − w_pos_z × |err_z|` | 1.0 / 1.5 | Stay near target `(x, y, z)` |
| **attitude** | `−w_attitude × (|err_roll| + |err_pitch|)` | 2.0 | Match target roll/pitch (usually 0) |
| **velocity** | `−w_lin_vel × ‖v‖ − w_ang_vel × ‖ω‖` | 0.15 / 0.08 | Damp motion |
| **upright_bonus** | `+w_upright_bonus × max(0, uprightness)` | 0.25 | Level flight |
| **terminal_penalty** | flip / crash / out_of_bounds | −50 / −30 / −40 | End episode on failure |

Hover termination: flip, crash (low z), out of bounds (XY drift), or time limit.

---

## Design notes

### Why progress + goal_dist?

- **Progress** gives credit only for **improvement**, which reduces reward for hovering near the goal without finishing.
- **goal_dist** adds a weak global pull toward the goal so the signal does not vanish when progress is noisy (wind, frame-to-frame jitter).

### Why penalize velocity if we want to reach the goal?

The agent must move, but unbounded speed increases crash and flip risk. Small `w_lin_vel` / `w_ang_vel` prefer controlled flight over aggressive zig-zagging.

### Relation to PPO

Stable-Baselines3 PPO maximizes **discounted return** `Σ γ^t r_t` with `gamma=0.99` by default in `train_nav_rl.py`. Terminal bonuses and penalties affect the value function over long horizons; dense progress terms provide credit assignment during the approach.

---

## Quick reference: navigation termination

```text
success     : ‖pos_xy − goal_xy‖ ≤ 0.32 m  AND  |pos_z − goal_z| ≤ 0.25 m
crash       : z ≤ 0.12 m (floor only)
unsafe tilt : tilt_deg ≥ SAFE_ATTITUDE_DEG → per-step penalty; optional episode end (NAV_END_ON_UNSAFE_ATTITUDE=1)
safe tilt   : tilt_deg < SAFE_ATTITUDE_DEG → per-step bonus
time_limit  : step ≥ 14_400 physics steps (~60 s)
```

Goal is sampled each episode at least `min_goal_distance` (2.5 m) from spawn, inside the map bounds (`±8 m` XY). Goal altitude is always **≥ spawn altitude**, uniformly in `[spawn_z, spawn_z + goal_z_max_rise]` (default spawn 1.25 m, rise up to 0.35 m).
