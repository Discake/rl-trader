"""Загрузка модели и прогон на тестовых данных."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from stable_baselines3 import A2C, DQN, PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.env import TradingEnv
from src.evaluation.metrics import compute_all_metrics
from src.training.trainer import make_env


_ALGO_CLASSES = {"ppo": PPO, "a2c": A2C, "sac": SAC, "td3": TD3, "dqn": DQN}


class Backtester:
    """Прогоняет обученную модель на тестовом фолде.

    Правильная загрузка:
      1. model.zip
      2. vec_normalize.pkl (статистики нормализации)
      3. тот же EnvConfig, что при обучении
    """

    def __init__(
        self,
        model_path: str | Path,
        vec_normalize_path: str | Path,
        algo_name: str,
        env_config,
    ) -> None:
        self.model_path = Path(model_path)
        self.vec_normalize_path = Path(vec_normalize_path)
        self.algo_name = algo_name
        self.env_config = env_config

        if algo_name not in _ALGO_CLASSES:
            raise ValueError(f"Неизвестный алгоритм: {algo_name!r}")

    def run(self, test_df) -> dict:
        """Прогон на test_df. Возвращает equity_curve, метрики и детали."""
        # 1. Собираем среду — та же, что при обучении, но на тестовых данных
        vec = DummyVecEnv([make_env(test_df, self.env_config)])

        # 2. Загружаем статистики нормализации
        vec = VecNormalize.load(str(self.vec_normalize_path), vec)
        vec.training = False           # статистики не обновлять
        vec.norm_reward = False        # награду не нормализовать

        # 3. Загружаем модель
        algo_cls = _ALGO_CLASSES[self.algo_name]
        model = algo_cls.load(str(self.model_path), env=vec)

        # 4. Прогон
        obs = vec.reset()
        equity = [self.env_config.initial_balance]
        done = False
        last_info: dict = {
            "portfolio_value": self.env_config.initial_balance,
            "balance": self.env_config.initial_balance,
            "positions": None,
            "total_fees": 0.0,
            "trade_count": 0,
        }

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = vec.step(action)
            done = bool(dones[0])
            info = infos[0]
            equity.append(info["portfolio_value"])
            last_info = info

        equity = np.array(equity, dtype=np.float64)
        n_trades = int(last_info.get("trade_count", 0))
        total_fees = float(last_info.get("total_fees", 0.0))

        metrics = compute_all_metrics(equity, n_trades, total_fees)

        final_positions = last_info.get("positions")
        if final_positions is None:
            final_positions = np.zeros(len(self.env_config.tickers), dtype=np.int64)

        return {
            "equity": equity,
            "metrics": metrics,
            "final_positions": np.asarray(final_positions),
            "final_balance": float(last_info.get("balance", 0.0)),
        }