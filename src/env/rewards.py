"""Функции награды для TradingEnv.

Каждая награда — это callable (prev_value, curr_value, info) -> float.
Опциональный метод reset() вызывается средой в начале каждого эпизода.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import Any

import numpy as np


class RewardFn(ABC):
    """Базовый класс функции награды."""

    @abstractmethod
    def __call__(
        self,
        prev_value: float,
        curr_value: float,
        info: dict[str, Any],
    ) -> float:
        ...

    def reset(self) -> None:
        """Вызывается в начале эпизода. По умолчанию ничего не делает."""
        return None


class PnLReward(RewardFn):
    """Простое изменение стоимости портфеля.

    Плюс: интуитивно понятно.
    Минус: агент не различает большой рост с высокой волатильностью
           и маленький стабильный рост.
    """

    def __call__(self, prev_value: float, curr_value: float, info: dict) -> float:
        return float(curr_value - prev_value)


class LogReturnReward(RewardFn):
    def __init__(
        self,
        scale: float = 1000.0,
        inactivity_penalty: float = 0.0,
        concentration_penalty: float = 0.0,
        turnover_penalty: float = 0.0,
    ):
        self.scale = scale
        self.inactivity_penalty = inactivity_penalty
        self.concentration_penalty = concentration_penalty
        self.turnover_penalty = turnover_penalty

    def __call__(self, prev_value, curr_value, info):
        if prev_value <= 0.0:
            return 0.0
        reward = self.scale * float(np.log(curr_value / prev_value))

        # 1. Штраф за бездействие
        positions = info.get("positions")
        if self.inactivity_penalty > 0 and positions is not None:
            if np.all(positions == 0):
                reward -= self.inactivity_penalty

        # 2. Штраф за концентрацию (HHI)
        prices = info.get("prices")
        if self.concentration_penalty > 0 and positions is not None and prices is not None:
            values = np.abs(positions * prices)
            total = values.sum()
            if total > 0:
                weights = values / total
                hhi = float((weights ** 2).sum())
                concentration = hhi - 1.0 / len(positions)
                reward -= self.concentration_penalty * concentration

        # 3. Штраф за turnover
        action = info.get("action")
        if self.turnover_penalty > 0 and action is not None:
            reward -= self.turnover_penalty * float(np.abs(action).sum())

        return reward


class SharpeReward(RewardFn):
    """Скользящее отношение доходности к волатильности.

    Агент учится зарабатывать стабильно, а не просто зарабатывать.
    Окно задаёт горизонт, на котором считается mean / std.
    """

    def __init__(self, window: int = 30, eps: float = 1e-8) -> None:
        self.window = window
        self.eps = eps
        self.returns: deque[float] = deque(maxlen=window)

    def reset(self) -> None:
        self.returns.clear()

    def __call__(self, prev_value: float, curr_value: float, info: dict) -> float:
        if prev_value <= 0.0:
            r = 0.0
        else:
            r = (curr_value - prev_value) / prev_value
        self.returns.append(r)

        if len(self.returns) < 2:
            return float(r)

        arr = np.asarray(self.returns, dtype=np.float64)
        std = float(arr.std())
        if std < self.eps:
            return float(arr.mean())
        return float(arr.mean() / std)


class CostAwareReward(RewardFn):
    """Лог-доходность с явным штрафом за издержки.

    В нашей среде комиссии уже вычтены из баланса при исполнении,
    так что дополнительно вычитать не нужно. Но если появятся модели,
    где это не так, здесь можно добавить штраф.
    """

    def __call__(self, prev_value: float, curr_value: float, info: dict) -> float:
        if prev_value <= 0.0:
            return 0.0
        return float(np.log(curr_value / prev_value))


# ---------- Фабрика ----------

def build_reward(name: str, params: dict[str, Any] | None = None) -> RewardFn:
    """Создаёт reward-функцию по имени из конфига."""
    params = params or {}
    if name == "pnl":
        return PnLReward()
    if name == "log_return":
        return LogReturnReward(**params)
    if name == "sharpe":
        return SharpeReward(**params)
    if name == "cost_aware":
        return CostAwareReward()
    raise ValueError(f"Неизвестный тип reward: {name!r}")