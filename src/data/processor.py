"""Расчёт технических индикаторов."""
from __future__ import annotations

import pandas as pd

from src.utils.config import DataConfig


class FeatureProcessor:
    """
    Добавляет технические индикаторы к OHLCV-данным.

    Все индикаторы причинно-обусловленные (causal):
    значение в момент t зависит только от данных до t включительно.
    Это исключает look-ahead bias.
    """

    def add_indicators(self, df: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
        result = df.copy()
        indicators = set(config.indicators)
        tickers = df.columns.get_level_values("ticker").unique()

        for ticker in tickers:
            close = df[(ticker, "close")]
            new_cols: dict[str, pd.Series] = {}

            if "rsi" in indicators:
                new_cols["rsi"] = self._rsi(close, config.rsi_window)

            if "macd" in indicators:
                macd, signal, hist = self._macd(
                    close, config.macd_fast, config.macd_slow, config.macd_signal
                )
                new_cols["macd"] = macd
                new_cols["macd_signal"] = signal
                new_cols["macd_hist"] = hist

            if "bb" in indicators:
                upper, mid, lower = self._bollinger(
                    close, config.bb_window, config.bb_std
                )
                new_cols["bb_upper"] = upper
                new_cols["bb_mid"] = mid
                new_cols["bb_lower"] = lower

            for name, series in new_cols.items():
                result[(ticker, name)] = series

        return result.sort_index(axis=1)

    # ---------- Индикаторы ----------

    @staticmethod
    def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
        avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
        rs = avg_gain / (avg_loss + 1e-12)
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def _macd(
        close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        hist = macd - signal_line
        return macd, signal_line, hist

    @staticmethod
    def _bollinger(
        close: pd.Series, window: int = 20, n_std: float = 2.0
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        mid = close.rolling(window).mean()
        std = close.rolling(window).std()
        upper = mid + n_std * std
        lower = mid - n_std * std
        return upper, mid, lower