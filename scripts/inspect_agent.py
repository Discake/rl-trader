"""Проверяет, что делает обученный агент на тесте."""
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from pathlib import Path
from src.data import DataLoader, FeatureProcessor, WalkForwardSplitter
from src.training.trainer import make_env
from src.utils.config import load_config


def main(run_dir:str|Path):
    config = load_config("configs/sac_daily.yaml")

    df = DataLoader(cache_dir=config.paths.data_raw).load(
        tickers=config.data.tickers, start=config.data.start,
        end=config.data.end, interval=config.data.interval,
    )
    df = FeatureProcessor().add_indicators(df, config.data)
    split = WalkForwardSplitter().split(df, config.split)[0]

    vec = DummyVecEnv([make_env(split.test, config.env)])
    vec = VecNormalize.load(f"{run_dir}/logs/vec_normalize.pkl", vec)
    vec.training = False
    vec.norm_reward = False

    model = SAC.load(f"{run_dir}/best_model/best_model", env=vec)

    obs = vec.reset()
    actions = []
    rewards = []
    values = []
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        actions.append(action[0].copy())
        obs, reward, dones, infos = vec.step(action)
        rewards.append(float(reward[0]))
        done = bool(dones[0])

    actions = np.array(actions)
    rewards = np.array(rewards)

    print(f"steps = {len(actions)}")
    print(f"\nДействия (сырые, до hmax):")
    print(f"  mean  = {actions.mean(axis=0)}")
    print(f"  std   = {actions.std(axis=0)}")
    print(f"  min   = {actions.min(axis=0)}")
    print(f"  max   = {actions.max(axis=0)}")
    print(f"  |a| < 0.01 (бездействие): {(np.abs(actions) < 0.01).mean() * 100:.1f}%")

    print(f"\nRewards:")
    print(f"  mean  = {rewards.mean():+.6f}")
    print(f"  std   = {rewards.std():.6f}")
    print(f"  sum   = {rewards.sum():+.6f}")
    print(f"  min   = {rewards.min():+.6f}")
    print(f"  max   = {rewards.max():+.6f}")


if __name__ == "__main__":
    main("experiments/sac/fold0_seed45")