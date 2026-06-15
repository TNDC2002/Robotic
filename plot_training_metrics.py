"""
Plot training metrics from ``metrics.csv`` written during PPO training.

Example:
  python plot_training_metrics.py runs/nav_ppo/metrics.csv
  python plot_training_metrics.py runs/nav_ppo/metrics.csv --out runs/nav_ppo/plots
  python plot_training_metrics.py runs/nav_ppo/metrics.csv --show
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def load_metrics(csv_path: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    rollout_rows: list[dict[str, str]] = []
    eval_rows: list[dict[str, str]] = []
    with csv_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            phase = row.get("phase", "").strip().lower()
            if phase == "eval":
                eval_rows.append(row)
            elif phase == "rollout":
                rollout_rows.append(row)
    return rollout_rows, eval_rows


def _series(rows: list[dict[str, str]], key: str) -> tuple[list[int], list[float]]:
    xs: list[int] = []
    ys: list[float] = []
    for row in rows:
        raw = row.get(key, "")
        if raw == "":
            continue
        xs.append(int(float(row["timesteps"])))
        ys.append(float(raw))
    return xs, ys


def plot_metrics(
    csv_path: Path,
    *,
    out_dir: Path | None = None,
    show: bool = False,
) -> list[Path]:
    rollout_rows, eval_rows = load_metrics(csv_path)
    if not rollout_rows and not eval_rows:
        raise SystemExit(f"No metrics rows found in {csv_path}")

    saved: list[Path] = []
    target_dir = out_dir or csv_path.parent / "plots"
    target_dir.mkdir(parents=True, exist_ok=True)

    def _plot_panel(
        filename: str,
        title: str,
        ylabel: str,
        value_key: str,
        *,
        prefer_eval: bool = False,
    ) -> None:
        fig, ax = plt.subplots(figsize=(9, 5))
        if rollout_rows and not prefer_eval:
            x, y = _series(rollout_rows, value_key)
            if x:
                ax.plot(x, y, label="rollout", alpha=0.55, linewidth=1.2)
        if eval_rows:
            x, y = _series(eval_rows, value_key)
            if x:
                ax.plot(x, y, label="eval", marker="o", linewidth=2.0)
        ax.set_title(title)
        ax.set_xlabel("timesteps")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        path = target_dir / filename
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(path)

    _plot_panel("mean_reward.png", "Mean episode reward", "reward", "mean_reward")
    _plot_panel(
        "success_rate.png",
        "Success rate",
        "fraction",
        "success_rate",
        prefer_eval=True,
    )
    _plot_panel("mean_ep_length.png", "Mean episode length", "steps", "mean_ep_length")

    if show:
        plt.show()
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot nav PPO metrics.csv")
    parser.add_argument(
        "csv",
        type=Path,
        nargs="?",
        default=Path("runs/nav_ppo/metrics.csv"),
        help="Path to metrics.csv (default: runs/nav_ppo/metrics.csv)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Directory for PNG files (default: <csv parent>/plots)",
    )
    parser.add_argument("--show", action="store_true", help="Open interactive plot windows")
    args = parser.parse_args()

    if not args.csv.is_file():
        raise SystemExit(f"Metrics file not found: {args.csv}")

    paths = plot_metrics(args.csv, out_dir=args.out, show=args.show)
    print(f"Wrote {len(paths)} plot(s) to {paths[0].parent}:")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
