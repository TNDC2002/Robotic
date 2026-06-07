"""Training callbacks for PPO navigation (Stable-Baselines3)."""

from __future__ import annotations

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback


class StopTrainingOnEvalPatience(BaseCallback):
    """Stop training when eval mean reward fails to improve for ``patience`` evaluations.

    Attach as ``callback_after_eval`` on ``EvalCallback``. Uses ``EvalCallback.last_mean_reward``
    after each evaluation round.
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
