"""Walk-forward × multi-seed эксперимент.

Для каждой комбинации (fold, seed):
  1. Создаёт свою директорию experiments/<name>/fold<i>_seed<s>/
  2. Обучает модель
  3. Оценивает на test-фолде
  4. Сохраняет metrics.json
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from src.data import DataLoader, FeatureProcessor, WalkForwardSplitter
from src.evaluation import Backtester
from src.training import Trainer
from src.utils.config import Config, load_config, save_config


def make_run_dir(base: Path, fold_idx: int, seed: int) -> Path:
    run_dir = base / f"fold{fold_idx}_seed{seed}"
    if run_dir.exists():
        shutil.rmtree(run_dir)   # чистим, чтобы не смешивать с прошлым запуском
    run_dir.mkdir(parents=True)
    return run_dir


def run_one(
    base_config: Config,
    train_df,
    val_df,
    test_df,
    run_dir: Path,
) -> dict:
    """Один запуск: обучение + оценка."""
    # 1. Сохраняем финальный конфиг в run_dir
    save_config(base_config, run_dir / "config.yaml")

    # 2. Обучение
    trainer = Trainer(base_config, train_df, val_df, run_dir=run_dir)
    trainer.train()

    # 3. Оценка
    bt = Backtester(
        model_path=run_dir / "best_model" / "best_model.zip",
        vec_normalize_path=run_dir / "logs" / "vec_normalize.pkl",
        algo_name=base_config.agent.name,
        env_config=base_config.env,
    )
    result = bt.run(test_df, save_to=run_dir / "metrics.json")
    return result["metrics"]


def main() -> None:
    base_config = load_config("configs/sac_daily.yaml")

    # Данные
    df = DataLoader(cache_dir=base_config.paths.data_raw).load(
        tickers=base_config.data.tickers,
        start=base_config.data.start,
        end=base_config.data.end,
        interval=base_config.data.interval,
    )
    df = FeatureProcessor().add_indicators(df, base_config.data)
    splits = WalkForwardSplitter().split(df, base_config.split)

    seeds = [42, 43, 44, 45, 46]
    experiment_dir = Path("experiments") / base_config.agent.name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    print(f"Эксперимент: {experiment_dir}")
    print(f"Фолдов: {len(splits)}, seeds: {seeds}")
    print(f"Всего запусков: {len(splits) * len(seeds)}")
    print()

    total = len(splits) * len(seeds)
    done = 0

    for fold_idx, split in enumerate(splits):
        for seed in seeds:
            done += 1
            run_dir = make_run_dir(experiment_dir, fold_idx, seed)
            print(f"[{done}/{total}] fold={fold_idx} seed={seed} → {run_dir}")

            # Копируем конфиг и меняем seed
            run_config = base_config.model_copy(update={"seed": seed})

            try:
                metrics = run_one(run_config, split.train, split.val, split.test, run_dir)
                print(
                    f"        return={metrics['total_return_pct']:+.2f}%  "
                    f"sharpe={metrics['sharpe']:+.3f}  "
                    f"maxDD={metrics['max_drawdown_pct']:+.2f}%  "
                    f"trades={metrics['n_trades']}"
                )
            except Exception as e:
                print(f"        [ERR] {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()