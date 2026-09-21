"""Загрузка и валидация конфигурации через Pydantic v2."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator


# ---------- Подмодели ----------

class DataConfig(BaseModel):
    tickers: list[str]
    start: str
    end: str
    interval: Literal["1d", "1h", "1m", "5m", "15m", "30m"] = "1d"
    indicators: list[str] = Field(default_factory=list)
    rsi_window: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_window: int = 20
    bb_std: float = 2.0

    @field_validator("tickers")
    @classmethod
    def tickers_not_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("tickers не может быть пустым")
        return v


class SplitConfig(BaseModel):
    method: Literal["walk_forward", "simple"] = "simple"
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    n_splits: int = 3
    test_size: float = 0.2


class EnvConfig(BaseModel):
    lookback: int = 30
    initial_balance: float = 100_000.0
    commission: float = 0.001
    slippage: float = 0.0005
    hmax: int = 100
    reward_type: Literal["pnl", "log_return", "sharpe", "cost_aware"] = "log_return"
    reward_params: dict[str, Any] = Field(default_factory=dict)


class AgentConfig(BaseModel):
    name: Literal["ppo", "a2c", "sac", "td3", "dqn"] = "ppo"
    policy: str = "MlpPolicy"
    learning_rate: float = 3e-4
    gamma: float = 0.99
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    clip_range: float = 0.2
    ent_coef: float = 0.0
    buffer_size: int = 100_000
    learning_starts: int = 1000
    tau: float = 0.005
    train_freq: int = 1
    gradient_steps: int = 1
    net_arch: list[int] = Field(default_factory=lambda: [64, 64])


class TrainingConfig(BaseModel):
    total_timesteps: int = 200_000
    eval_freq: int = 5000
    n_eval_episodes: int = 5
    save_freq: int = 10_000
    early_stopping_patience: int = 10


class NormalizeConfig(BaseModel):
    norm_obs: bool = True
    norm_reward: bool = True
    clip_obs: float = 10.0


class PathsConfig(BaseModel):
    data_raw: str = "data/raw"
    data_processed: str = "data/processed"
    logs: str = "logs"
    checkpoints: str = "checkpoints"
    best_model: str = "best_model"
    tensorboard: str = "tb_logs"


# ---------- Корневая модель ----------

class Config(BaseModel):
    seed: int = 42
    device: Literal["cpu", "cuda", "auto"] = "cpu"
    data: DataConfig
    split: SplitConfig = Field(default_factory=SplitConfig)
    env: EnvConfig = Field(default_factory=EnvConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    normalize: NormalizeConfig = Field(default_factory=NormalizeConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)


# ---------- Загрузка ----------

def _deep_merge(base: dict, override: dict) -> dict:
    """Рекурсивно мержит override в base (override побеждает)."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Конфиг не найден: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(experiment_path: str | Path,
                default_path: str | Path = "configs/default.yaml") -> Config:
    """
    Загружает default.yaml, мержит поверх экспериментальный конфиг,
    валидирует через Pydantic и возвращает Config.

    Если experiment_path == default_path, просто грузит один файл.
    """
    experiment_path = Path(experiment_path)
    default_path = Path(default_path)

    if experiment_path.resolve() == default_path.resolve():
        merged = _load_yaml(experiment_path)
    else:
        base = _load_yaml(default_path)
        override = _load_yaml(experiment_path)
        merged = _deep_merge(base, override)

    return Config(**merged)


def save_config(config: Config, path: str | Path) -> None:
    """Сохраняет финальный конфиг (после мержа и валидации) — для воспроизводимости."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config.model_dump(), f, sort_keys=False, allow_unicode=True)