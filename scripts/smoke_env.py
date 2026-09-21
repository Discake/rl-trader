"""Smoke test: прогон среды на случайных действиях."""
import numpy as np

from src.data import DataLoader, FeatureProcessor, WalkForwardSplitter
from src.env import TradingEnv
from src.utils.config import load_config


def main(seed:int=42) -> None:
    config = load_config("configs/ppo_daily.yaml")

    # данные
    df = DataLoader(cache_dir=config.paths.data_raw).load(
        tickers=config.data.tickers,
        start=config.data.start,
        end=config.data.end,
        interval=config.data.interval,
    )
    df = FeatureProcessor().add_indicators(df, config.data)
    split = WalkForwardSplitter().split(df, config.split)[0]

    # среда
    env = TradingEnv(split.train, config.env)
    print(f"stock_dim       = {env.stock_dim}")
    print(f"n_features      = {env.n_features}")
    print(f"observation_dim = {env.observation_space.shape[0]}")
    print(f"action_dim      = {env.action_space.shape[0]}")

    env = TradingEnv(split.train, config.env)
    env.action_space.seed(seed)
    obs, _ = env.reset(seed=seed)
    
    total_reward = 0.0
    steps = 0
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

    prices = env.prices_array
    first = prices[env.start_day]
    last = prices[-1]
    bh_return = ((last / first).mean() - 1) * 100
    print(f"Buy & hold return: {bh_return:.2f}%")
    
    return {
        "seed": seed,
        "steps": steps,
        "total_reward": total_reward,
        "final_value": info["portfolio_value"],
        "final_balance": info["balance"],
        "positions": info["positions"].tolist(),
    }


if __name__ == "__main__":
    # прогон с разными seed — убедиться, что среда ведёт себя стабильно
    r = main()
    print(r)