"""Сравнение агента (случайного или обученного) со стратегией buy & hold."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import DataLoader, FeatureProcessor, WalkForwardSplitter
from src.env import TradingEnv
from src.utils.config import load_config


# ----------------------------------------------------------------------
# Метрики
# ----------------------------------------------------------------------

def compute_metrics(equity: np.ndarray, n_trades: int = 0,
                    total_fees: float = 0.0) -> dict:
    """Считает метрики по кривой капитала."""
    equity = np.asarray(equity, dtype=np.float64)
    returns = np.diff(equity) / equity[:-1]
    returns = returns[np.isfinite(returns)]

    if len(returns) == 0:
        return {
            "total_return_pct": 0.0,
            "sharpe": 0.0,
            "max_drawdown_pct": 0.0,
            "volatility_pct": 0.0,
            "n_trades": n_trades,
            "total_fees": total_fees,
        }

    total_return = equity[-1] / equity[0] - 1.0
    sharpe = returns.mean() / (returns.std() + 1e-12) * np.sqrt(252)
    vol = returns.std() * np.sqrt(252)

    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    max_dd = drawdown.min()

    return {
        "total_return_pct": total_return * 100,
        "sharpe": sharpe,
        "max_drawdown_pct": max_dd * 100,
        "volatility_pct": vol * 100,
        "n_trades": n_trades,
        "total_fees": total_fees,
    }


# ----------------------------------------------------------------------
# Стратегии
# ----------------------------------------------------------------------

def run_random_agent(env: TradingEnv, seed: int = 42) -> tuple[np.ndarray, int, float]:
    """Прогон эпизода со случайными действиями.

    Возвращает (equity_curve, n_trades, total_fees).
    """
    env.action_space.seed(seed)
    obs, _ = env.reset(seed=seed)

    equity = [env.config.initial_balance]
    n_trades = 0
    total_fees = 0.0
    done = False

    while not done:
        action = env.action_space.sample()

        # считаем издержки до шага, чтобы потом оценить
        balance_before = env.balance
        positions_before = env.positions.copy()

        obs, reward, done, truncated, info = env.step(action)

        # число изменений позиций = число сделок
        delta_positions = np.abs(env.positions - positions_before).sum()
        n_trades += int(delta_positions)

        equity.append(info["portfolio_value"])

    total_fees = info["total_fees"]

    # издержки: разница между «идеальным» и реальным балансом
    # проще оценить как сумму commission + slippage, применённых к сделкам
    # здесь мы не храним их явно, поэтому пересчитаем приблизительно
    # (для точности нужно логировать в среде — см. ниже)
    return np.array(equity), n_trades, total_fees


def run_buy_and_hold(df: pd.DataFrame, tickers: list[str],
                     initial_balance: float,
                     commission: float, slippage: float) -> tuple[np.ndarray, int, float]:
    """Buy & hold: на первом шаге покупаем равные доли каждого актива.

    Покупка по цене close первого дня + slippage + commission.
    Дальше ничего не делаем, просто считаем стоимость портфеля каждый день.
    """
    prices = np.array([df[(t, "close")].to_numpy(dtype=np.float64) for t in tickers]).T
    n_days, n_assets = prices.shape

    # на первом шаге делим баланс поровну
    per_asset = initial_balance / n_assets
    positions = np.zeros(n_assets, dtype=np.int64)
    balance = initial_balance

    # покупка на первый день
    buy_price = prices[0] * (1.0 + slippage)
    cost_per_share = buy_price * (1.0 + commission)
    for i in range(n_assets):
        shares = int(per_asset // cost_per_share[i])
        positions[i] = shares
        balance -= shares * cost_per_share[i]

    # equity curve
    equity = balance + (prices @ positions)

    # издержки: сколько заплатили сверх «чистой» цены
    clean_cost = (prices[0] * positions).sum()
    actual_cost = initial_balance - balance
    total_fees = actual_cost - clean_cost

    return equity, n_assets, float(total_fees)


# ----------------------------------------------------------------------
# Вывод
# ----------------------------------------------------------------------

def print_comparison(results: dict[str, dict]) -> None:
    """Печатает таблицу сравнения."""
    metrics = ["total_return_pct", "sharpe", "max_drawdown_pct",
               "volatility_pct", "n_trades", "total_fees"]
    headers = ["metric"] + list(results.keys())

    # ширины колонок
    widths = [max(len(h), max(len(m) for m in metrics)) for h in headers]
    for i, name in enumerate(results.keys(), start=1):
        widths[i] = max(widths[i], max(len(f"{results[name][m]:.4f}") for m in metrics))

    # шапка
    line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    print(line)
    print("-" * len(line))

    # строки
    for m in metrics:
        row = [m.ljust(widths[0])]
        for i, name in enumerate(results.keys(), start=1):
            v = results[name][m]
            if m in ("n_trades",):
                s = f"{int(v)}"
            elif m == "total_fees":
                s = f"{v:.2f}"
            else:
                s = f"{v:.4f}"
            row.append(s.ljust(widths[i]))
        print("  ".join(row))


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def main() -> None:
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

    print(f"Данные: {split.train.index.min().date()} → "
          f"{split.train.index.max().date()} ({len(split.train)} дней)")
    print(f"Тикеры: {config.data.tickers}")
    print(f"Стартовый капитал: {config.env.initial_balance:,.2f}")
    print()

    results: dict[str, dict] = {}

    # 1. Buy & hold
    equity_bh, n_trades_bh, fees_bh = run_buy_and_hold(
        split.train, config.data.tickers,
        config.env.initial_balance,
        config.env.commission,
        config.env.slippage,
    )
    results["buy_and_hold"] = compute_metrics(equity_bh, n_trades_bh, fees_bh)

    # 2. Случайный агент
    env = TradingEnv(split.train, config.env)
    equity_rnd, n_trades_rnd, fees_rnd = run_random_agent(env, seed=config.seed)
    results["random_agent"] = compute_metrics(equity_rnd, n_trades_rnd, fees_rnd)

    # вывод
    print_comparison(results)

    # финальные значения
    print()
    print(f"Final value (buy & hold):    {equity_bh[-1]:>12,.2f}")
    print(f"Final value (random agent):  {equity_rnd[-1]:>12,.2f}")


if __name__ == "__main__":
    main()