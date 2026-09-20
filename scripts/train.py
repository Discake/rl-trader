from src.utils.config import load_config, save_config

config = load_config("configs/ppo_daily.yaml")
print(config.agent.name)          # "ppo"
print(config.env.reward_type)     # "sharpe" (переопределено в эксперименте)
print(config.data.tickers)        # ["AAPL", "MSFT", "GOOGL"]

# Сохраняем финальный конфиг рядом с моделью — для воспроизводимости
save_config(config, f"{config.paths.logs}/run_config.yaml")