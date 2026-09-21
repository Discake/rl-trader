"""Фабрика RL-агентов на базе Stable Baselines3.

Разделяет гиперпараметры на три группы:
- общие (policy, learning_rate, gamma, net_arch, ...)
- on-policy (n_steps, n_epochs, clip_range, ent_coef) — для PPO/A2C
- off-policy (buffer_size, learning_starts, tau, train_freq, ...) — для SAC/TD3/DQN
"""
from __future__ import annotations

from typing import Any

import torch
from stable_baselines3 import A2C, DQN, PPO, SAC, TD3
from stable_baselines3.common.base_class import BaseAlgorithm

from src.utils.config import AgentConfig


# Реестр алгоритмов
ALGOS: dict[str, type[BaseAlgorithm]] = {
    "ppo": PPO,
    "a2c": A2C,
    "sac": SAC,
    "td3": TD3,
    "dqn": DQN,
}

# Какие алгоритмы поддерживают непрерывные действия
CONTINUOUS_ALGOS = {"ppo", "a2c", "sac", "td3"}
# Какие — только дискретные
DISCRETE_ONLY_ALGOS = {"dqn"}


def _resolve_device(device: str) -> str:
    """Разрешает 'auto' в конкретное устройство."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _build_policy_kwargs(config: AgentConfig) -> dict[str, Any]:
    """Превращает net_arch в формат SB3."""
    return {
        "net_arch": list(config.net_arch),
        "activation_fn": torch.nn.Tanh,
    }


def _build_common_kwargs(config: AgentConfig, device: str) -> dict[str, Any]:
    """Параметры, общие для всех алгоритмов."""
    return {
        "policy": config.policy,
        "learning_rate": config.learning_rate,
        "gamma": config.gamma,
        "policy_kwargs": _build_policy_kwargs(config),
        "device": device,
        "verbose": 1,
    }


def _build_on_policy_kwargs(config: AgentConfig) -> dict[str, Any]:
    """Параметры PPO и A2C."""
    kwargs: dict[str, Any] = {
        "n_steps": config.n_steps,
        "ent_coef": config.ent_coef,
    }
    if config.name == "ppo":
        kwargs["batch_size"] = config.batch_size
        kwargs["n_epochs"] = config.n_epochs
        kwargs["clip_range"] = config.clip_range
    return kwargs


def _build_off_policy_kwargs(config: AgentConfig) -> dict[str, Any]:
    """Параметры SAC, TD3 и DQN."""
    kwargs: dict[str, Any] = {
        "buffer_size": config.buffer_size,
        "learning_starts": config.learning_starts,
        "train_freq": config.train_freq,
        "gradient_steps": config.gradient_steps,
        "batch_size": config.batch_size,
    }
    if config.name in ("sac", "td3"):
        kwargs["tau"] = config.tau
    return kwargs


def build_agent(
    config: AgentConfig,
    env,
    device: str = "cpu",
    tensorboard_log: str | None = None,
    seed: int | None = None,
) -> BaseAlgorithm:
    if config.name not in ALGOS:
        raise ValueError(
            f"Неизвестный алгоритм: {config.name!r}. "
            f"Доступные: {sorted(ALGOS.keys())}"
        )

    algo_cls = ALGOS[config.name]
    device = _resolve_device(device)

    kwargs: dict[str, Any] = {}
    kwargs.update(_build_common_kwargs(config, device))

    if config.name in ("ppo", "a2c"):
        kwargs.update(_build_on_policy_kwargs(config))
    elif config.name in ("sac", "td3", "dqn"):
        kwargs.update(_build_off_policy_kwargs(config))

    if tensorboard_log is not None:
        kwargs["tensorboard_log"] = tensorboard_log
    if seed is not None:
        kwargs["seed"] = seed

    return algo_cls(env=env, **kwargs)


def list_algos() -> list[str]:
    return sorted(ALGOS.keys())


def is_continuous(name: str) -> bool:
    return name in CONTINUOUS_ALGOS


def is_discrete_only(name: str) -> bool:
    return name in DISCRETE_ONLY_ALGOS