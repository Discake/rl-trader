import gymnasium as gym  # <-- Важно: используем gymnasium, а не устаревший gym
import gym_anytrading
from gym_anytrading.envs import StocksEnv  # <-- Правильный импорт класса
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Загружаем данные (встроенный датасет Google)
df = gym_anytrading.datasets.STOCKS_GOOGL.copy()

# Создаем среду через gym.make (рекомендуемый способ)
# Обратите внимание: параметры df, window_size и frame_bound передаются как аргументы
env = gym.make('stocks-v0', df=df, window_size=10, frame_bound=(10, len(df)))

# Оборачиваем для Stable Baselines3
env = DummyVecEnv([lambda: env])

# Создаем и обучаем агента PPO
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)

print("Обучение прошло успешно!")