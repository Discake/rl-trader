"""Оценка обученной модели на test-фолде и сравнение с baseline."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data import DataLoader, FeatureProcessor, WalkForwardSplitter
from src.env import TradingEnv
from src.evaluation import Backtester, compute_all_metrics, format_metrics_table
from src.evaluation.backtest import _ALGO_CLASSES
from src.utils.config import load_config


def buy_and_hold_equity(df, tickers, initial_balance, commission, slippage):
    prices = np.array([df[(t, "close")].to_numpy(dtype=np.float64) for t in tickers]).T
    n_assets = prices.shape[1]
    per_asset = initial_balance / n_assets

    positions = np.zeros(n_assets, dtype=np.int64)
    balance = initial_balance
    buy_price = prices[0] * (1.0 + slippage)
    cost_per_share = buy_price * (1.0 + commission)

    for i in range(n_assets):
        shares = int(per_asset // cost_per_share[i])
        positions[i] = shares
        balance -= shares * cost_per_share[i]

    equity = balance + prices @ positions
    return equity


def run_random_baseline(test_df, env_config, seed=42):
    env = TradingEnv(test_df, env_config)
    env.action_space.seed(seed)
    obs, _ = env.reset(seed=seed)
    equity = [env_config.initial_balance]
    done = False
    while not done:
        action = env.action_space.sample()
        obs, _, done, truncated, info = env.step(action)
        equity.append(info["portfolio_value"])
    return np.array(equity), env.trade_count, env.total_fees


def main() -> None:
    config = load_config("configs/ppo_daily.yaml")

    # Данные
    df = DataLoader(cache_dir=config.paths.data_raw).load(
        tickers=config.data.tickers,
        start=config.data.start,
        end=config.data.end,
        interval=config.data.interval,
    )
    df = FeatureProcessor().add_indicators(df, config.data)
    split = WalkForwardSplitter().split(df, config.split)[0]

    test_df = split.test
    print(f"Test: {test_df.index.min().date()} → {test_df.index.max().date()} "
          f"({len(test_df)} дней)\n")

    results: dict[str, dict] = {}

    # 1. Buy & hold
    bh_equity = buy_and_hold_equity(
        test_df, config.data.tickers,
        config.env.initial_balance,
        config.env.commission,
        config.env.slippage,
    )
    results["buy_hold"] = compute_all_metrics(bh_equity, n_trades=len(config.data.tickers), total_fees=0.0)

    # 2. Random
    rnd_equity, rnd_trades, rnd_fees = run_random_baseline(
        test_df, config.env, seed=config.seed,
    )
    results["random"] = compute_all_metrics(rnd_equity, rnd_trades, rnd_fees)

    # 3. Обученная модель
    best_model_path = Path(config.paths.best_model) / "best_model.zip"
    vec_norm_path = Path(config.paths.logs) / "vec_normalize.pkl"

    if best_model_path.exists() and vec_norm_path.exists():
        bt = Backtester(
            model_path=best_model_path,
            vec_normalize_path=vec_norm_path,
            algo_name=config.agent.name,
            env_config=config.env,
        )
        out = bt.run(test_df)
        results["rl_agent"] = out["metrics"]
    else:
        print(f"[WARN] Модель не найдена: {best_model_path}")
        print(f"[WARN] VecNormalize не найден: {vec_norm_path}")

    # Вывод
    print(format_metrics_table(results))

    # Дополнительно: разница между агентом и buy & hold
    if "rl_agent" in results:
        print("\nРазница с buy & hold:")
        for m in ["total_return_pct", "sharpe", "max_drawdown_pct"]:
            diff = results["rl_agent"][m] - results["buy_hold"][m]
            sign = "+" if diff > 0 else ""
            print(f"  {m:20s} {sign}{diff:.4f}")

    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    plt.plot(bh_equity, label="Buy & Hold", linewidth=1.5)
    plt.plot(rnd_equity, label="Random", linewidth=1.0, alpha=0.6)
    if "rl_agent" in results:
        out = bt.run(test_df)  # или сохранить из предыдущего вызова
        plt.plot(out["equity"], label="RL Agent", linewidth=1.8)
    plt.legend()
    plt.title("Equity Curve — Test Fold")
    plt.xlabel("Step")
    plt.ylabel("Portfolio Value")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("logs/equity_curve.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()