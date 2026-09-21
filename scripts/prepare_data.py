"""Скачивает данные, добавляет индикаторы, делает split, сохраняет результат."""
from pathlib import Path

import pandas as pd

from src.data import DataLoader, FeatureProcessor, WalkForwardSplitter
from src.utils.config import load_config


def main() -> None:
    config = load_config("configs/ppo_daily.yaml")

    # 1. Загрузка
    loader = DataLoader(cache_dir=config.paths.data_raw)
    df = loader.load(
        tickers=config.data.tickers,
        start=config.data.start,
        end=config.data.end,
        interval=config.data.interval,
    )
    print(f"Загружено: {df.shape[0]} строк, {df.shape[1]} колонок")
    print(f"Период: {df.index.min().date()} → {df.index.max().date()}")

    # 2. Индикаторы
    processor = FeatureProcessor()
    df = processor.add_indicators(df, config.data)
    print(f"После индикаторов: {df.shape[1]} колонок")

    # 3. Split
    splitter = WalkForwardSplitter()
    splits = splitter.split(df, config.split)
    print(f"Фолдов: {len(splits)}")
    for i, s in enumerate(splits):
        print(
            f"  Fold {i + 1}: train={len(s.train)}, "
            f"val={len(s.val)}, test={len(s.test)}"
        )

    # 4. Сохранение первого фолда для проверки
    out_dir = Path(config.paths.data_processed)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "full_processed.parquet")
    print(f"Сохранено в {out_dir / 'full_processed.parquet'}")


if __name__ == "__main__":
    main()