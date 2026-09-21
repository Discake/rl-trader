"""Собирает все metrics.json в одну таблицу и считает статистику."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


def parse_run_id(name: str) -> tuple[int, int] | None:
    m = re.match(r"fold(\d+)_seed(\d+)", name)
    if m is None:
        return None
    return int(m.group(1)), int(m.group(2))


def collect(experiment_dir: Path) -> pd.DataFrame:
    rows = []
    for run_dir in sorted(experiment_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        parsed = parse_run_id(run_dir.name)
        if parsed is None:
            continue
        fold_idx, seed = parsed

        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue

        with metrics_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        metrics = data["metrics"]
        metrics["fold"] = fold_idx
        metrics["seed"] = seed
        metrics["run_id"] = run_dir.name
        rows.append(metrics)

    return pd.DataFrame(rows)


def print_summary(df: pd.DataFrame) -> None:
    metric_cols = ["total_return_pct", "sharpe", "sortino",
                   "max_drawdown_pct", "volatility_pct", "calmar",
                   "n_trades", "total_fees", "final_value"]

    print("=" * 80)
    print("ОБЩАЯ СТАТИСТИКА ПО ВСЕМ ЗАПУСКАМ")
    print("=" * 80)
    print(df[metric_cols].describe().round(4).to_string())

    print()
    print("=" * 80)
    print("ПО ФОЛДАМ (средние значения)")
    print("=" * 80)
    per_fold = df.groupby("fold")[metric_cols].mean().round(4)
    print(per_fold.to_string())

    print()
    print("=" * 80)
    print("ПО SEED (средние значения)")
    print("=" * 80)
    per_seed = df.groupby("seed")[metric_cols].mean().round(4)
    print(per_seed.to_string())

    print()
    print("=" * 80)
    print("СТАБИЛЬНОСТЬ (std по seed внутри каждого фолда)")
    print("=" * 80)
    stability = df.groupby("fold")[["sharpe", "total_return_pct"]].std().round(4)
    print(stability.to_string())


def main() -> None:
    experiment_dir = Path("experiments/sac")   # поменяйте на ppo если нужно
    if not experiment_dir.exists():
        print(f"Нет директории {experiment_dir}")
        return

    df = collect(experiment_dir)
    if df.empty:
        print("Нет данных для агрегации")
        return

    # Сохраняем полную таблицу
    df.to_csv(experiment_dir / "summary.csv", index=False)
    print(f"Сохранено: {experiment_dir / 'summary.csv'}")
    print()

    print_summary(df)


if __name__ == "__main__":
    main()