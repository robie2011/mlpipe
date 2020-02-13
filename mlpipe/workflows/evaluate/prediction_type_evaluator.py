from abc import ABC, abstractmethod
from typing import Dict

import numpy as np


class PredictionTypeEvaluator(ABC):
    @abstractmethod
    def evaluate(self, predictions: np.ndarray, targets: np.ndarray) -> Dict:
        pass

    @abstractmethod
    def prediction_formatter(self, predictions: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def mse(predictions: np.ndarray, targets: np.ndarray) -> float:
        return np.sqrt(((predictions - targets) ** 2).mean())
