"""Training Layer: обучение агентов с callbacks и нормализацией."""
from .callbacks import EarlyStoppingCallback, NaNDetectorCallback
from .trainer import Trainer, make_env

__all__ = [
    "Trainer",
    "make_env",
    "EarlyStoppingCallback",
    "NaNDetectorCallback",
]