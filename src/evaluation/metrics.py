"""Метрики для оценки торговых стратегий."""
from __future__ import annotations

import numpy as np


def equity_to_returns(equity: np.ndarray) -> np.ndarray:
    """Доходности по кривой капитала."""
    equity = np.asarray(equity, dtype=np.float64)
    returns = np.diff(equity) / equity[:-1]
    return returns[np.isfinite(returns)]


def total_return(equity: np.ndarray) -> float:
    """Итоговая доходность в процентах."""
    equity = np.asarray(equity, dtype=np.float64)
    return float((equity[-1] / equity[0] - 1.0) * 100)


def sharpe_ratio(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """Годовой коэффициент Шарпа (без risk-free rate)."""
    returns = np.asarray(returns, dtype=np.float64)
    if len(returns) < 2:
        return 0.0
    std = returns.std()
    if std < 1e-12:
        return 0.0
    return float(returns.mean() / std * np.sqrt(periods_per_year))


def sortino_ratio(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """Sortino: учитывает только downside-волатильность."""
    returns = np.asarray(returns, dtype=np.float64)
    if len(returns) < 2:
        return 0.0
    downside = returns[returns < 0]
    if len(downside) < 2 or downside.std() < 1e-12:
        return 0.0
    return float(returns.mean() / downside.std() * np.sqrt(periods_per_year))


def max_drawdown(equity: np.ndarray) -> float:
    """Максимальная просадка в процентах (отрицательное число)."""
    equity = np.asarray(equity, dtype=np.float64)
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    return float(drawdown.min() * 100)


def volatility(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """Годовая волатильность в процентах."""
    returns = np.asarray(returns, dtype=np.float64)
    if len(returns) < 2:
        return 0.0
    return float(returns.std() * np.sqrt(periods_per_year) * 100)


def calmar_ratio(equity: np.ndarray, periods_per_year: int = 252) -> float:
    """Calmar: годовая доходность / max drawdown."""
    equity = np.asarray(equity, dtype=np.float64)
    if len(equity) < 2:
        return 0.0
    total = equity[-1] / equity[0] - 1.0
    n_years = len(equity) / periods_per_year
    annual_return = (1.0 + total) ** (1.0 / max(n_years, 1e-9)) - 1.0
    mdd = abs(max_drawdown(equity)) / 100
    if mdd < 1e-12:
        return 0.0
    return float(annual_return / mdd)


def compute_all_metrics(
    equity: np.ndarray,
    n_trades: int = 0,
    total_fees: float = 0.0,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """Полный набор метрик одним вызовом."""
    returns = equity_to_returns(equity)
    return {
        "total_return_pct": total_return(equity),
        "sharpe": sharpe_ratio(returns, periods_per_year),
        "sortino": sortino_ratio(returns, periods_per_year),
        "max_drawdown_pct": max_drawdown(equity),
        "volatility_pct": volatility(returns, periods_per_year),
        "calmar": calmar_ratio(equity, periods_per_year),
        "n_trades": n_trades,
        "total_fees": total_fees,
        "final_value": float(equity[-1]),
    }


def format_metrics_table(results: dict[str, dict]) -> str:
    """Форматирует словарь метрик в ASCII-таблицу."""
    metrics = [
        "total_return_pct", "sharpe", "sortino", "max_drawdown_pct",
        "volatility_pct", "calmar", "n_trades", "total_fees", "final_value",
    ]
    columns = list(results.keys())

    # ширины
    col_widths = {c: max(len(c), 14) for c in columns}
    metric_width = max(len(m) for m in metrics)

    # шапка
    header = "metric".ljust(metric_width) + "  " + "  ".join(
        c.rjust(col_widths[c]) for c in columns
    )
    sep = "-" * len(header)

    lines = [header, sep]
    for m in metrics:
        row = m.ljust(metric_width) + "  "
        for c in columns:
            v = results[c].get(m, 0.0)
            if m == "n_trades":
                s = f"{int(v)}"
            elif m == "total_fees":
                s = f"{v:,.2f}"
            elif m == "final_value":
                s = f"{v:,.2f}"
            else:
                s = f"{v:+.4f}" if "drawdown" in m or "return" in m else f"{v:.4f}"
            row += s.rjust(col_widths[c]) + "  "
        lines.append(row.rstrip())
    return "\n".join(lines)