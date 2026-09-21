"""Обучение RL-агента на исторических данных."""
from src.data import DataLoader, FeatureProcessor, WalkForwardSplitter
from src.training import Trainer
from src.utils.config import load_config


def main() -> None:
    config = load_config("configs/sac_daily.yaml")

    # 1. Данные
    df = DataLoader(cache_dir=config.paths.data_raw).load(
        tickers=config.data.tickers,
        start=config.data.start,
        end=config.data.end,
        interval=config.data.interval,
    )
    df = FeatureProcessor().add_indicators(df, config.data)

    # 2. Разбиение — берём первый фолд
    splits = WalkForwardSplitter().split(df, config.split)
    split = splits[0]

    print(f"Train: {split.train.index.min().date()} → {split.train.index.max().date()} "
          f"({len(split.train)} дней)")
    print(f"Val:   {split.val.index.min().date()} → {split.val.index.max().date()} "
          f"({len(split.val)} дней)")
    print(f"Test:  {split.test.index.min().date()} → {split.test.index.max().date()} "
          f"({len(split.test)} дней)")

    # 3. Обучение
    trainer = Trainer(config, split.train, split.val)
    trainer.train()


if __name__ == "__main__":
    main()