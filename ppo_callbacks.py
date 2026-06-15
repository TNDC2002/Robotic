"""Training callbacks for PPO navigation (Stable-Baselines3)."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

METRICS_CSV_COLUMNS = (
    "phase",
    "timesteps",
    "mean_reward",
    "std_reward",
    "mean_ep_length",
    "success_rate",
    "n_episodes",
)


class StopTrainingOnEvalPatience(BaseCallback):
    """Stop training when eval mean reward fails to improve for ``patience`` evaluations.

    Pass to ``EvalCallback(..., callback_after_eval=...)`` — SB3 stores that as
    ``EvalCallback.callback`` and runs it after each eval. Assigning
    ``eval_cb.callback_after_eval`` after construction does not wire it up.
    """

    parent: EvalCallback

    def __init__(
        self,
        patience: int,
        *,
        min_delta: float = 0.0,
        min_evals: int = 1,
        verbose: int = 1,
    ):
        super().__init__(verbose=verbose)
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")
        if min_evals < 1:
            raise ValueError(f"min_evals must be >= 1, got {min_evals}")
        self.patience = patience
        self.min_delta = min_delta
        self.min_evals = min_evals
        self._best_mean_reward = -np.inf
        self._evals_without_improvement = 0
        self._n_evals = 0

    def _on_step(self) -> bool:
        assert self.parent is not None, "StopTrainingOnEvalPatience must be used with EvalCallback"
        mean_reward = float(self.parent.last_mean_reward)
        self._n_evals += 1

        if mean_reward > self._best_mean_reward + self.min_delta:
            if self.verbose >= 1 and self._n_evals > 1:
                print(
                    f"Eval improved: {self._best_mean_reward:.2f} -> {mean_reward:.2f} "
                    f"(min_delta={self.min_delta})"
                )
            self._best_mean_reward = mean_reward
            self._evals_without_improvement = 0
        else:
            self._evals_without_improvement += 1
            if self.verbose >= 1:
                print(
                    f"No eval improvement ({self._evals_without_improvement}/{self.patience}), "
                    f"best={self._best_mean_reward:.2f}, last={mean_reward:.2f}"
                )

        if self._n_evals >= self.min_evals and self._evals_without_improvement >= self.patience:
            if self.verbose >= 1:
                print(
                    f"Early stopping after {self._n_evals} evals: "
                    f"no improvement for {self.patience} consecutive evaluations "
                    f"(best mean reward {self._best_mean_reward:.2f})"
                )
            return False
        return True


class NavEvalCallback(EvalCallback):
    """EvalCallback that also tracks success rate and episode-length stats from the nav env."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_count = 0
        self.last_std_reward = 0.0
        self.last_mean_ep_length = 0.0
        self.last_success_rate = 0.0

    def _drain_eval_outcomes(self) -> list[dict]:
        try:
            nested = self.eval_env.env_method("drain_completed_outcomes")
        except (AttributeError, TypeError):
            return []
        return [item for sublist in nested for item in sublist]

    def _on_step(self) -> bool:
        continue_training = super()._on_step()
        outcomes = self._drain_eval_outcomes()
        if not outcomes:
            return continue_training

        self.eval_count += 1
        reasons = [o.get("terminal_reason") for o in outcomes]
        self.last_success_rate = sum(1 for r in reasons if r == "success") / len(outcomes)

        if self.evaluations_results:
            last_rewards = self.evaluations_results[-1]
            self.last_std_reward = float(np.std(last_rewards))
        if self.evaluations_length:
            self.last_mean_ep_length = float(np.mean(self.evaluations_length[-1]))
        return continue_training


class MetricsCsvCallback(BaseCallback):
    """Append rollout and eval metrics to a CSV file for offline plotting."""

    def __init__(
        self,
        csv_path: Path | str,
        *,
        eval_callback: NavEvalCallback | None = None,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.csv_path = Path(csv_path)
        self.eval_callback = eval_callback
        self._last_logged_eval = 0
        self._header_written = False

    def _on_training_start(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
            self._write_header()

    def _write_header(self) -> None:
        with self.csv_path.open("w", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=METRICS_CSV_COLUMNS).writeheader()
        self._header_written = True

    def _append_row(self, row: dict[str, object]) -> None:
        if not self._header_written and not self.csv_path.exists():
            self._write_header()
        with self.csv_path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=METRICS_CSV_COLUMNS)
            writer.writerow({key: row.get(key, "") for key in METRICS_CSV_COLUMNS})

    def _on_rollout_end(self) -> None:
        buffer = list(self.model.ep_info_buffer)
        if not buffer:
            return

        rewards = [float(ep["r"]) for ep in buffer]
        lengths = [float(ep["l"]) for ep in buffer]
        reasons = [ep.get("terminal_reason") for ep in buffer if ep.get("terminal_reason")]
        success_rate = ""
        if reasons:
            success_rate = sum(1 for r in reasons if r == "success") / len(reasons)

        self._append_row(
            {
                "phase": "rollout",
                "timesteps": int(self.num_timesteps),
                "mean_reward": float(np.mean(rewards)),
                "std_reward": float(np.std(rewards)),
                "mean_ep_length": float(np.mean(lengths)),
                "success_rate": success_rate,
                "n_episodes": len(buffer),
            }
        )

    def _on_step(self) -> bool:
        if self.eval_callback is None:
            return True
        if self.eval_callback.eval_count <= self._last_logged_eval:
            return True

        self._last_logged_eval = self.eval_callback.eval_count
        self._append_row(
            {
                "phase": "eval",
                "timesteps": int(self.num_timesteps),
                "mean_reward": float(self.eval_callback.last_mean_reward),
                "std_reward": float(self.eval_callback.last_std_reward),
                "mean_ep_length": float(self.eval_callback.last_mean_ep_length),
                "success_rate": float(self.eval_callback.last_success_rate),
                "n_episodes": eval_settings_episodes(self.eval_callback),
            }
        )
        return True


def eval_settings_episodes(eval_callback: EvalCallback) -> int:
    """Best-effort episode count for the last eval round."""
    if eval_callback.evaluations_results:
        return len(eval_callback.evaluations_results[-1])
    return getattr(eval_callback, "n_eval_episodes", 0)
