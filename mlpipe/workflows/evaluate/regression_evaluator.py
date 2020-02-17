from typing import Dict

import numpy as np

from mlpipe.workflows.evaluate.prediction_type_evaluator import PredictionTypeEvaluator


class RegressionEvaluator(PredictionTypeEvaluator):
    def evaluate(self, predictions: np.ndarray, targets: np.ndarray) -> Dict:
        result = {'mse': self.mse(predictions=predictions[:, 0], targets=targets)}
        return result

    def prediction_formatter(self, predictions: np.ndarray) -> np.ndarray:
        return predictions
