# RL Trader

**RL Trader** — исследовательский Python-проект для обучения и оценки reinforcement learning агентов на исторических рыночных данных. Проект сочетает подготовку OHLCV, технические индикаторы, собственную среду Gymnasium, обучение с Stable-Baselines3 и бэктестинг с базовыми стратегиями.

> **Ветки:** [`main`](https://github.com/Discake/rl-trader/tree/main) содержит базовый пайплайн; активная работа над экспериментами идёт в [`development`](https://github.com/Discake/rl-trader/tree/development). Для скриптов multi-seed и walk-forward экспериментов смотрите именно `development`. Наличие реализации не означает, что эксперименты прошли валидацию на реальном рынке.

## Возможности

- Загрузка OHLCV через `yfinance` с Parquet-кэшем.
- Расчёт индикаторов (RSI, MACD, Bollinger Bands) и разбиение на train/validation/test, включая walk-forward.
- Кастомная Gymnasium-среда с моделированием стоимости портфеля, комиссии и проскальзывания.
- Фабрика агентов PPO, A2C, SAC, TD3 и DQN на Stable-Baselines3. **Ограничение:** DQN требует дискретных действий; текущая торговая среда использует непрерывное пространство действий.
- Варианты reward: PnL, log return, скользящая Sharpe-подобная оценка, cost-aware. Не следует считать их доказанной реализацией differential Sharpe ratio.
- Обучение с сохранением моделей, checkpoint, нормализации и конфигураций.
- Сравнение со стратегиями buy-and-hold и random по доходности, Sharpe, просадке и волатильности.
- **Дополнительно в `development`:** `run_experiment.py` (фолды × seed), `run_seeds.py` и `aggregate_results.py` для сводной статистики. Эти сценарии требуют дополнительной проверки воспроизводимости.

## Стек

Python 3.10+ · NumPy · pandas · yfinance · Gymnasium · Stable-Baselines3 · PyTorch · Pydantic · PyYAML · Matplotlib · Parquet

## Быстрый старт

Из корня проекта:

```bash
git clone https://github.com/Discake/rl-trader.git
cd rl-trader
git switch development
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e .
```

**Важно:** `pyproject.toml` пока не фиксирует сторонние зависимости, поэтому `pip install -e .` недостаточно для запуска. Нужно дополнительно установить библиотеки, используемые модулями (в том числе `numpy`, `pandas`, `yfinance`, `pyarrow`, библиотеку индикаторов из `src/data/processor.py`, `gymnasium`, `stable-baselines3`, `torch`, `pydantic`, `pyyaml`, `matplotlib`). Совместимость их версий не проверялась при написании README.

Основные команды (проверяйте зависимости и конфигурации перед запуском):

```bash
python -m scripts.prepare_data
python -m scripts.train
python -m scripts.evaluate
python -m scripts.compare_baselines
```

В ветке `development` также предусмотрены:

```bash
python -m scripts.run_experiment
python -m scripts.run_seeds
python -m scripts.aggregate_results
```

Скрипты используют пути конфигураций, заданные в исходном коде: `configs/default.yaml`, `configs/ppo_daily.yaml`, `configs/sac_daily.yaml`. Они не принимают универсальный параметр `--config`. **Внимание:** экспериментальные скрипты удаляют ранее существующую директорию конкретного запуска `foldN_seedM` перед повторным запуском.

## Структура

- `src/data` — загрузка и обработка данных, разбиение на фолды.
- `src/env` — торговая среда и reward.
- `src/agents` — фабрика алгоритмов.
- `src/training` — цикл обучения и callback.
- `src/evaluation` — backtest и метрики.
- `scripts` — подготовка данных, обучение, оценка и эксперименты.
- `configs` — базовые настройки и overrides.

## Статус и ограничения

Это **исследовательский прототип**, а не торговый робот для реальных счетов. В репозитории нет брокерской интеграции. Файлы тестов в `tests/` пока пустые; наличие тестовой директории не означает наличие работающего набора тестов. Корректность оценок зависит от отсутствия утечек данных, реалистичности издержек и качества исторических данных. Ни одна заявка на доходность или инвестиционную эффективность здесь не делается.

[Исходники базовой версии](https://github.com/Discake/rl-trader/tree/main) · [Ветка разработки и эксперименты](https://github.com/Discake/rl-trader/tree/development)

*Educational and research purposes only. Not investment advice.*
