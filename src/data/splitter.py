"""Разбиение данных на train/val/test с защитой от look-ahead bias."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.utils.config import SplitConfig


@dataclass(frozen=True)
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


class WalkForwardSplitter:
    """
    Разбивает данные на несколько последовательных фолдов.

    Схема (rolling, окна не пересекаются):
        Fold 1: [---train---][-val-][-test-]
        Fold 2:                             [---train---][-val-][-test-]
        Fold 3:                                                         ...

    Каждый фолд покрывает свой временной период — это проверяет
    устойчивость стратегии к разным рыночным режимам.
    """

    def split(self, df: pd.DataFrame, config: SplitConfig) -> list[Split]:
        if config.method == "simple":
            return [self._simple_split(df, config)]
        if config.method == "walk_forward":
            return self._walk_forward(df, config)
        raise ValueError(f"Неизвестный метод разбиения: {config.method}")

    # ---------- Простое разбиение ----------

    def _simple_split(self, df: pd.DataFrame, config: SplitConfig) -> Split:
        n = len(df)
        train_end = int(n * config.train_ratio)
        val_end = int(n * (config.train_ratio + config.val_ratio))

        if val_end >= n:
            raise ValueError(
                f"train_ratio + val_ratio = {config.train_ratio + config.val_ratio} "
                f"оставляет < 1 строки под test"
            )

        return Split(
            train=df.iloc[:train_end].copy(), # Бро, тебе надо тренироваться © S1mple
            val=df.iloc[train_end:val_end].copy(),
            test=df.iloc[val_end:].copy(),
        )

    # ---------- Walk-forward ----------

    def _walk_forward(self, df: pd.DataFrame, config: SplitConfig) -> list[Split]:
        n = len(df)
        window_size = n // config.n_splits
        splits: list[Split] = []

        for i in range(config.n_splits):
            start = i * window_size
            end = start + window_size if i < config.n_splits - 1 else n
            window = df.iloc[start:end]
            splits.append(self._simple_split(window, config))

        return splits