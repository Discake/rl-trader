"""Загрузка OHLCV-данных через yfinance с локальным кэшем."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf


class DataLoader:
    """
    Скачивает OHLCV-данные и кэширует их в parquet.

    Возвращаемый формат:
    - Индекс: DatetimeIndex (даты)
    - Колонки: MultiIndex (ticker, field), field ∈ {open, high, low, close, volume}
    """

    def __init__(self, cache_dir: str | Path = "data/raw") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load(self, tickers, start, end, interval="1d", use_cache=True):
        tickers = sorted(set(tickers))
        cache_file = self._cache_path(tickers, start, end, interval)
        print(f"[DEBUG] CWD = {Path.cwd()}")
        print(f"[DEBUG] cache_dir = {self.cache_dir.resolve()}")
        print(f"[DEBUG] cache_file = {cache_file.resolve()}")
        print(f"[DEBUG] cache exists before = {cache_file.exists()}")

        if use_cache and cache_file.exists():
            print("[DEBUG] using cache")
            return pd.read_parquet(cache_file)

        raw = self._download(tickers, start, end, interval)
        df = self._normalize(raw, tickers)
        print(f"[DEBUG] df shape = {df.shape}")

        cache_file.parent.mkdir(parents=True, exist_ok=True)  # на всякий случай
        df.to_parquet(cache_file)
        print(f"[DEBUG] cache written, exists after = {cache_file.exists()}")
        return df

    def _download(
        self,
        tickers: list[str],
        start: str,
        end: str,
        interval: str,
    ) -> pd.DataFrame:
        return yf.download(
            tickers=tickers,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=True,      # скорректированные цены (дивиденды, сплиты)
            progress=False,
            group_by="ticker",     # колонки -> (ticker, field)
            threads=True,
        )

    def _normalize(self, raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
        """Приводим формат yfinance к единому MultiIndex (ticker, field)."""
        fields = ["Open", "High", "Low", "Close", "Volume"]

        # Один тикер: yfinance возвращает плоские колонки
        if not isinstance(raw.columns, pd.MultiIndex):
            if len(tickers) != 1:
                raise ValueError("Ожидался MultiIndex для нескольких тикеров")
            raw.columns = pd.MultiIndex.from_product(
                [[tickers[0]], raw.columns], names=["ticker", "field"]
            )
        else:
            raw.columns.names = ["ticker", "field"]

        # Оставляем только нужные поля
        raw = raw.loc[:, (slice(None), fields)].copy()

        # Приводим имена полей к нижнему регистру
        rename = {f: f.lower() for f in fields}
        raw.columns = pd.MultiIndex.from_tuples(
            [(t, rename[f]) for t, f in raw.columns],
            names=["ticker", "field"],
        )

        # Убираем полностью пустые строки (праздники), сортируем
        raw = raw.dropna(how="all").sort_index()

        return raw

    def _cache_path(self, tickers: list[str], start: str, end: str, interval: str) -> Path:
        key = f"{'-'.join(tickers)}_{start}_{end}_{interval}".replace("/", "-")
        return self.cache_dir / f"{key}.parquet"