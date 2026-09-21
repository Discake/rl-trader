"""Trainer: собирает среду, агента, callbacks и запускает обучение."""
from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.agents import build_agent
from src.env import TradingEnv
from src.training.callbacks import EarlyStoppingCallback, NaNDetectorCallback
from src.utils.config import Config


def make_env(df, env_config):
    """Фабрика среды для DummyVecEnv."""
    def _init():
        env = TradingEnv(df, env_config)
        env = Monitor(env)
        return env
    return _init


class Trainer:
    """Обучает RL-агента по конфигу и сохраняет артефакты."""

    def __init__(self, config: Config, train_df, val_df):
        self.config = config
        self.train_df = train_df
        self.val_df = val_df

        self.paths = self._prepare_paths()
        self.train_env = self._build_train_env()
        self.eval_env = self._build_eval_env()
        self.agent = build_agent(
            config.agent,
            self.train_env,
            device=config.device,
            tensorboard_log=str(self.paths["tensorboard"]),
            seed=config.seed,
        )

    # ------------------------------------------------------------------
    # Подготовка
    # ------------------------------------------------------------------

    def _prepare_paths(self) -> dict[str, Path]:
        p = self.config.paths
        paths = {
            "logs": Path(p.logs),
            "checkpoints": Path(p.checkpoints),
            "best_model": Path(p.best_model),
            "tensorboard": Path(p.tensorboard),
        }
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)
        return paths

    def _build_train_env(self):
        vec = DummyVecEnv([make_env(self.train_df, self.config.env)])
        vec = VecNormalize(
            vec,
            norm_obs=self.config.normalize.norm_obs,
            norm_reward=self.config.normalize.norm_reward,
            clip_obs=self.config.normalize.clip_obs,
        )
        return vec

    def _build_eval_env(self):
        vec = DummyVecEnv([make_env(self.val_df, self.config.env)])
        vec = VecNormalize(
            vec,
            norm_obs=self.config.normalize.norm_obs,
            norm_reward=False,          # при оценке награду НЕ нормализуем
            clip_obs=self.config.normalize.clip_obs,
            training=False,             # статистики не обновляются
        )
        return vec

    # ------------------------------------------------------------------
    # Обучение
    # ------------------------------------------------------------------

    def _build_callbacks(self) -> CallbackList:
        training_cfg = self.config.training

        eval_cb = EvalCallback(
            self.eval_env,
            best_model_save_path=str(self.paths["best_model"]),
            log_path=str(self.paths["logs"]),
            eval_freq=training_cfg.eval_freq,
            n_eval_episodes=training_cfg.n_eval_episodes,
            deterministic=True,
            render=False,
            verbose=1,
        )

        checkpoint_cb = CheckpointCallback(
            save_freq=training_cfg.save_freq,
            save_path=str(self.paths["checkpoints"]),
            name_prefix="rl_trader",
            verbose=0,
        )

        early_stop_cb = EarlyStoppingCallback(
            patience=training_cfg.early_stopping_patience,
            verbose=1,
        )

        nan_cb = NaNDetectorCallback(check_freq=5000, verbose=0)

        # Порядок важен: EarlyStopping читает результаты EvalCallback,
        # поэтому EvalCallback должен выполниться первым.
        return CallbackList([eval_cb, checkpoint_cb, early_stop_cb, nan_cb])

    def train(self) -> None:
        """Запускает обучение."""
        # Синхронизируем статистики нормализации eval-среды с train-средой
        self.eval_env.obs_rms = self.train_env.obs_rms
        self.eval_env.ret_rms = self.train_env.ret_rms

        self.agent.learn(
            total_timesteps=self.config.training.total_timesteps,
            callback=self._build_callbacks(),
            progress_bar=True,
        )

        # финальное сохранение
        self._save()

    # ------------------------------------------------------------------
    # Сохранение
    # ------------------------------------------------------------------

    def _save(self) -> None:
        """Сохраняет финальную модель, статистики нормализации и конфиг."""
        from src.utils.config import save_config

        model_path = self.paths["logs"] / "final_model"
        self.agent.save(str(model_path))

        # VecNormalize — обязательно, иначе при загрузке статистики потеряются
        self.train_env.save(str(self.paths["logs"] / "vec_normalize.pkl"))

        # конфиг — для воспроизводимости
        save_config(self.config, self.paths["logs"] / "config.yaml")
        print(f"Артефакты сохранены в {self.paths['logs']}")