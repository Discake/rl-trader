"""Smoke test: создаём агента каждого типа и делаем один шаг обучения."""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.agents import build_agent, list_algos
from src.utils.config import load_config


class DummyEnv(gym.Env):
    """Простейшая среда для проверки фабрики."""
    def __init__(self):
        self.observation_space = spaces.Box(-1, 1, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Box(-1, 1, shape=(2,), dtype=np.float32)
        self._step = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step = 0
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        self._step += 1
        obs = np.zeros(4, dtype=np.float32)
        reward = float(-np.sum(action ** 2))
        done = self._step >= 20
        return obs, reward, done, False, {}


class DummyDiscreteEnv(DummyEnv):
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(3)


def main() -> None:
    config = load_config("configs/ppo_daily.yaml")

    for name in list_algos():
        # DQN требует дискретных действий
        if name == "dqn":
            env = DummyDiscreteEnv()
        else:
            env = DummyEnv()

        # копируем конфиг и меняем имя алгоритма
        agent_cfg = config.agent.model_copy(update={"name": name})

        try:
            agent = build_agent(
                agent_cfg, env,
                device="cpu",
                seed=config.seed,
            )
            # 1 итерация обучения
            agent.learn(total_timesteps=64, progress_bar=False)
            print(f"[OK]  {name:4s}  собрался и обучился на 64 шагах")
        except Exception as e:
            print(f"[ERR] {name:4s}  {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()