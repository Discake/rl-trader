"""TradingEnv: gymnasium-совместимая среда для RL-трейдинга."""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from src.env.rewards import RewardFn, build_reward
from src.utils.config import EnvConfig


class TradingEnv(gym.Env):
    """Торговая среда для одного или нескольких активов.

    Формат данных: MultiIndex DataFrame (ticker, field), где field ∈
    {open, high, low, close, volume} + опциональные индикаторы.

    Состояние (state):
        [balance, positions..., prices..., features...]

    Действие (action):
        Box(-1, 1, shape=(stock_dim,)) — нормализованный размер сделки.
        Умножается на hmax внутри среды.

    Награда (reward):
        Зависит от выбранной reward_fn.
    """

    metadata = {"render_modes": []}

    def __init__(self, df: pd.DataFrame, config: EnvConfig) -> None:
        super().__init__()

        if len(df) < config.lookback + 2:
            raise ValueError(
                f"Данных слишком мало: {len(df)} строк, "
                f"нужно минимум {config.lookback + 2}"
            )

        self.df = df
        self.config = config

        # тикеры и признаки
        self.tickers: list[str] = sorted(
            df.columns.get_level_values("ticker").unique().tolist()
        )
        self.stock_dim = len(self.tickers)

        base_fields = {"open", "high", "low", "close", "volume"}
        self.feature_cols: list[str] = [
            c for c in df.columns.get_level_values("field").unique()
            if c not in base_fields
        ]
        self.n_features = len(self.feature_cols)

        # предвычисляем массивы для быстрого step()
        self.prices_array = self._build_prices_array()          # (T, stock_dim)
        self.features_array = self._build_features_array()      # (T, stock_dim * n_features)

        # пространства
        state_dim = 1 + self.stock_dim * (2 + self.n_features)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.stock_dim,),
            dtype=np.float32,
        )

        # reward
        self.reward_fn: RewardFn = build_reward(
            config.reward_type, config.reward_params
        )

        # границы эпизода
        self.start_day = config.lookback
        self.max_day = len(df) - 1

        # в __init__ после определения feature_cols
        assert not np.isnan(self.features_array[self.start_day]).any(), \
            "lookback слишком мал: в первом состоянии есть NaN"

        # изменяемые поля (инициализируются в reset)
        self.day: int = self.start_day
        self.balance: float = config.initial_balance
        self.positions: np.ndarray = np.zeros(self.stock_dim, dtype=np.int64)

        self.total_fees: float = 0.0
        self.trade_count: int = 0

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        self.day = self.start_day
        self.balance = float(self.config.initial_balance)
        self.positions = np.zeros(self.stock_dim, dtype=np.int64)
        self.reward_fn.reset()

        self.total_fees = 0.0
        self.trade_count = 0

        return self._get_state(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        # 1. ограничиваем действие
        action = np.clip(action, -1.0, 1.0).astype(np.float64)

        # 2. текущие цены и стоимость портфеля ДО сделки
        prices_t = self.prices_array[self.day]
        prev_value = self.balance + float(np.dot(self.positions, prices_t))

        # 3. исполняем сделку
        self._execute_action(action, prices_t)

        # 4. переходим на следующий день
        self.day += 1
        done = self.day >= self.max_day
        truncated = False

        # 5. стоимость портфеля ПОСЛЕ сдвига цен
        prices_next = self.prices_array[self.day]
        curr_value = self.balance + float(np.dot(self.positions, prices_next))

        # 6. reward
        info = {
            "portfolio_value": curr_value,
            "balance": self.balance,
            "positions": self.positions.copy(),
            "day": self.day,
            "total_fees": self.total_fees,
            "trade_count": self.trade_count,
            "prices": prices_next.copy(),    # ← добавьте
            "action": action.copy(),          # ← добавьте
        }
        reward = self.reward_fn(prev_value, curr_value, info)

        return self._get_state(), float(reward), done, truncated, info

    # ------------------------------------------------------------------
    # Внутренние методы
    # ------------------------------------------------------------------

    def _execute_action(self, action: np.ndarray, prices: np.ndarray) -> None:
        """Исполняет действие с учётом комиссии и проскальзывания."""
        desired_shares = (action * self.config.hmax).astype(np.int64)
        commission = self.config.commission
        slippage = self.config.slippage

        for i, delta in enumerate(desired_shares):
            if delta == 0:
                continue
            price = float(prices[i])
            if price <= 0.0:
                continue

            if delta > 0:
                buy_price = price * (1.0 + slippage)
                cost_per_share = buy_price * (1.0 + commission)
                max_shares = int(self.balance // cost_per_share)
                shares = min(int(delta), max_shares)
                if shares > 0:
                    self.trade_count += 1
                    clean_cost = shares * price
                    actual_cost = shares * cost_per_share
                    self.total_fees += actual_cost - clean_cost
                    self.balance -= actual_cost
                    self.positions[i] += shares
            else:
                sell_price = price * (1.0 - slippage)
                revenue_per_share = sell_price * (1.0 - commission)
                shares = min(int(-delta), int(self.positions[i]))
                if shares > 0:
                    self.trade_count += 1
                    clean_revenue = shares * price
                    actual_revenue = shares * revenue_per_share
                    self.total_fees += clean_revenue - actual_revenue
                    self.balance += actual_revenue
                    self.positions[i] -= shares

    def _get_state(self) -> np.ndarray:
        prices = self.prices_array[self.day]
        features = self.features_array[self.day]
        state = np.concatenate(
            [
                np.array([self.balance], dtype=np.float64),
                self.positions.astype(np.float64),
                prices,
                features,
            ]
        )
        return state.astype(np.float32)

    def _build_prices_array(self) -> np.ndarray:
        return np.array(
            [self.df[(t, "close")].to_numpy(dtype=np.float64) for t in self.tickers]
        ).T  # (T, stock_dim)

    def _build_features_array(self) -> np.ndarray:
        if not self.feature_cols:
            return np.zeros((len(self.df), 0), dtype=np.float64)
        cols = []
        for ticker in self.tickers:
            for feature in self.feature_cols:
                cols.append(self.df[(ticker, feature)].to_numpy(dtype=np.float64))
        return np.stack(cols, axis=1)  # (T, stock_dim * n_features)