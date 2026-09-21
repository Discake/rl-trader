"""Дополнительные callbacks поверх SB3."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class EarlyStoppingCallback(BaseCallback):
    """Останавливает обучение, если eval-награда не улучшается N проверок подряд."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4, verbose: int = 0):
        super().__init__(verbose)
        self.patience = patience
        self.min_delta = min_delta
        self.best_mean_reward = -np.inf
        self.counter = 0

    def _on_step(self) -> bool:
        # EvalCallback пишет результаты в self.locals после каждой оценки
        if "eval_mean_reward" not in self.locals:
            return True

        current = self.locals["eval_mean_reward"]
        if current > self.best_mean_reward + self.min_delta:
            self.best_mean_reward = current
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"[EarlyStopping] без улучшения: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                if self.verbose:
                    print(f"[EarlyStopping] стоп на шаге {self.num_timesteps}")
                return False
        return True


class NaNDetectorCallback(BaseCallback):
    """Падает с ошибкой, если в политике или value-сети появились NaN.

    В финансовых средах NaN часто возникает из-за плохой нормализации
    или деления на ноль в reward — лучше узнать об этом сразу.
    """

    def __init__(self, check_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq != 0:
            return True

        # SB3 хранит policy в self.model.policy
        for name, param in self.model.policy.named_parameters():
            if not param.requires_grad:
                continue
            if not torch_isfinite(param).all():
                raise RuntimeError(
                    f"NaN/inf обнаружен в {name} на шаге {self.num_timesteps}"
                )
        return True


def torch_isfinite(t):
    import torch
    return torch.isfinite(t)