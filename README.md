# RL Trader

A research-oriented reinforcement learning project for evaluating trading agents on historical market data. It provides a configurable pipeline for fetching prices, computing technical indicators, splitting datasets, training agents, and comparing backtest results with simple baselines.

**Status:** experimental research code, not a production trading system. Backtesting does not establish future profitability. The repository currently has no pinned runtime dependencies and the test files under `tests/` are empty.

## What is implemented

- **Market data:** OHLCV data download through `yfinance`, normalized to a consistent pandas MultiIndex format and cached as Parquet.
- **Features:** configurable technical indicators, including RSI, MACD and Bollinger Bands.
- **Data splits:** simple and walk-forward splitting, with separate training, validation, and test portions.
- **Trading simulation:** a custom Gymnasium environment with portfolio state, configurable transaction commission and slippage.
- **Agents:** Stable-Baselines3 factory for PPO, A2C, SAC, TD3 and DQN. Note that DQN requires a discrete action space; the trading environment is designed around continuous actions, so it is not automatically compatible with DQN.
- **Rewards:** PnL, log return, rolling Sharpe-like reward and cost-aware variant. These are experimental reward formulations; the code should not be described as a validated differential Sharpe implementation.
- **Training:** evaluation callbacks, checkpoints, early stopping, NaN detection, normalization and saved model artifacts.
- **Evaluation:** comparisons with buy-and-hold and random-action baselines, including return, Sharpe ratio, volatility, maximum drawdown and trade statistics.

## Stack

Python 3.10+, pandas, NumPy, yfinance, Gymnasium, Stable-Baselines3, PyTorch, Pydantic v2, PyYAML, Matplotlib, TensorBoard (optional), Parquet support (such as pyarrow).

## Setup

Run from the repository root:

```bash
git clone https://github.com/Discake/rl-trader.git
cd rl-trader
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install numpy pandas yfinance pyarrow pandas-ta gymnasium stable-baselines3 torch pydantic pyyaml matplotlib tensorboard tqdm rich
```

> The dependency list above is indicative, not a verified lockfile. Some technical-indicator library versions may have compatibility constraints. Check `src/data/processor.py` and adjust installation for your platform as needed.

## Usage

The repository's entry points use specific configuration files (rather than taking CLI configuration flags):

```bash
# Download historical data, compute indicators and save a parquet dataset
python -m scripts.prepare_data

# Train the configured SAC agent (uses configs/sac_daily.yaml)
python -m scripts.train

# Evaluate on the held-out test partition, compare baselines and plot equity
python -m scripts.evaluate

# Compare buy-and-hold and random policy in the training partition
python -m scripts.compare_baselines

# Smoke-check algorithm construction
python -m scripts.smoke_env
```

The default configuration lives at `configs/default.yaml`, with experiment-specific overrides in `configs/ppo_daily.yaml` and `configs/sac_daily.yaml`. To change the configuration used by an entry point, edit its `load_config(...)` call.

**Outputs:** historical data is cached in `data/raw/`; processed data in `data/processed/`; training artifacts are written under `logs/`, `checkpoints/`, `best_model/` and `tb_logs/` according to configuration. The evaluation script expects the trained best model at `best_model/best_model.zip` and normalization statistics at `logs/vec_normalize.pkl`.

## Repository layout

```text
configs/           Default and experiment-specific YAML configuration
scripts/           Data preparation, training, evaluation, baselines and smoke tests
src/agents/        Algorithm factory
src/data/          Data loading, features and train/validation/test splitting
src/env/           Trading environment and reward functions
src/evaluation/    Backtesting and performance metrics
src/training/      Training workflow and callbacks
src/utils/         Configuration loading and validation
tests/             Test placeholders (not yet implemented)
```

## Limitations

- Research/backtesting only; no broker execution integration.
- Data quality, look-ahead bias and realistic transaction costs require careful validation.
- The main training script uses the **first** walk-forward fold; it does not run full cross-validation automatically.
- Tests currently contain placeholders; no passing test suite is claimed.
- Training outcomes, backtest performance and dependency compatibility have **not** been independently verified for this README.

## Disclaimer

For educational and research purposes only. Not investment advice.
