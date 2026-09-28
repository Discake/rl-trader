"""Обучение SAC с несколькими seed на одном фолде.

Меняете SEEDS и FOLD_INDICES — получаете нужный срез эксперимента.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import DataLoader, FeatureProcessor, WalkForwardSplitter
from src.evaluation import Backtester
from src.training import Trainer
from src.utils.config import load_config, save_config


# ----------------------------------------------------------------------
# Настройки эксперимента
# ----------------------------------------------------------------------

SEEDS = [47]        # добавьте сюда больше, когда захотите
FOLD_INDICES = [0]          # [0, 1, 2] — для полного walk-forward
EXPERIMENT_NAME = "sac_3assets"     # должен совпадать с agent.name в конфиге
CONFIG_PATH = "configs/sac_daily.yaml"


# ----------------------------------------------------------------------
# Запуск одного прогона
# ----------------------------------------------------------------------

def make_run_dir(base: Path, fold_idx: int, seed: int) -> Path:
    run_dir = base / f"fold{fold_idx}_seed{seed}"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    return run_dir


def run_one(config, train_df, val_df, test_df, run_dir: Path) -> dict:
    # фиксируем итоговый конфиг запуска
    save_config(config, run_dir / "config.yaml")

    # обучение
    trainer = Trainer(config, train_df, val_df, run_dir=run_dir)
    trainer.train()

    # оценка на тесте
    bt = Backtester(
        model_path=run_dir / "best_model" / "best_model.zip",
        vec_normalize_path=run_dir / "logs" / "vec_normalize.pkl",
        algo_name=config.agent.name,
        env_config=config.env,
    )
    result = bt.run(test_df, save_to=run_dir / "metrics.json")
    return result["metrics"]


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def main() -> None:
    base_config = load_config(CONFIG_PATH)

    # Данные
    df = DataLoader(cache_dir=base_config.paths.data_raw).load(
        tickers=base_config.data.tickers,
        start=base_config.data.start,
        end=base_config.data.end,
        interval=base_config.data.interval,
    )
    print(df.isna().sum())
    df = FeatureProcessor().add_indicators(df, base_config.data)
    splits = WalkForwardSplitter().split(df, base_config.split)

    experiment_dir = Path("experiments") / EXPERIMENT_NAME
    experiment_dir.mkdir(parents=True, exist_ok=True)

    total = len(FOLD_INDICES) * len(SEEDS)
    print(f"Эксперимент: {experiment_dir}")
    print(f"Фолды: {FOLD_INDICES}, seeds: {SEEDS}")
    print(f"Всего запусков: {total}")
    print()

    rows: list[dict] = []
    done = 0

    for fold_idx in FOLD_INDICES:
        split = splits[fold_idx]
        for seed in SEEDS:
            done += 1
            run_dir = make_run_dir(experiment_dir, fold_idx, seed)
            print(f"[{done}/{total}] fold={fold_idx}  seed={seed}")

            run_config = base_config.model_copy(update={"seed": seed})

            try:
                metrics = run_one(
                    run_config, split.train, split.val, split.test, run_dir,
                )
                row = dict(metrics)
                row["fold"] = fold_idx
                row["seed"] = seed
                rows.append(row)

                print(
                    f"     return={metrics['total_return_pct']:+.2f}%  "
                    f"sharpe={metrics['sharpe']:+.3f}  "
                    f"maxDD={metrics['max_drawdown_pct']:+.2f}%  "
                    f"trades={metrics['n_trades']}  "
                    f"fees={metrics['total_fees']:.2f}"
                )
            except Exception as e:
                print(f"     [ERR] {type(e).__name__}: {e}")

    # Сохраняем сводку
    if rows:
        summary = pd.DataFrame(rows)
        summary.to_csv(experiment_dir / "summary.csv", index=False)
        print(f"\nСводка сохранена: {experiment_dir / 'summary.csv'}")
        print_stability(summary)


def print_stability(df: pd.DataFrame) -> None:
    metric_cols = [
        "total_return_pct", "sharpe", "sortino",
        "max_drawdown_pct", "volatility_pct", "calmar",
        "n_trades", "total_fees", "final_value",
    ]
    available = [c for c in metric_cols if c in df.columns]

    print("\n" + "=" * 72)
    print("СТАБИЛЬНОСТЬ ПО SEED (mean ± std)")
    print("=" * 72)

    for c in available:
        vals = df[c].to_numpy(dtype=np.float64)
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        mn = float(np.min(vals))
        mx = float(np.max(vals))

        if c == "n_trades":
            print(f"  {c:20s}  mean={mean:8.1f}  std={std:6.1f}  min={mn:6.0f}  max={mx:6.0f}")
        elif c == "total_fees":
            print(f"  {c:20s}  mean={mean:8.2f}  std={std:6.2f}  min={mn:6.2f}  max={mx:6.2f}")
        elif c == "final_value":
            print(f"  {c:20s}  mean={mean:12,.2f}  std={std:8.2f}  min={mn:12,.2f}  max={mx:12,.2f}")
        else:
            print(f"  {c:20s}  mean={mean:+8.4f}  std={std:6.4f}  min={mn:+8.4f}  max={mx:+8.4f}")

    # Проверки устойчивости
    print("\n" + "=" * 72)
    print("ПРОВЕРКИ УСТОЙЧИВОСТИ")
    print("=" * 72)

    checks = []
    if "sharpe" in df.columns:
        sharpe = df["sharpe"].to_numpy()
        checks.append((
            "Все seed с положительным Sharpe",
            bool((sharpe > 0).all()),
            f"min={sharpe.min():+.3f}",
        ))
        checks.append((
            "Sharpe std < 1.0",
            bool(sharpe.std(ddof=1) < 1.0) if len(sharpe) > 1 else True,
            f"std={sharpe.std(ddof=1):.3f}" if len(sharpe) > 1 else "n/a",
        ))

    if "total_return_pct" in df.columns:
        ret = df["total_return_pct"].to_numpy()
        checks.append((
            "Все seed с положительной доходностью",
            bool((ret > 0).all()),
            f"min={ret.min():+.2f}%",
        ))

    if "max_drawdown_pct" in df.columns:
        dd = df["max_drawdown_pct"].to_numpy()
        checks.append((
            "Max drawdown < 10% по модулю",
            bool((dd > -10.0).all()),
            f"worst={dd.min():+.2f}%",
        ))

    for name, ok, detail in checks:
        mark = "[OK] " if ok else "[!!] "
        print(f"  {mark}{name:45s}  {detail}")


if __name__ == "__main__":
    main()