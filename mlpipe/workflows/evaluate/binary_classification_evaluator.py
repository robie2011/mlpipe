from typing import Dict

import numpy as np
from sklearn.metrics import confusion_matrix

from mlpipe.workflows.evaluate.prediction_type_evaluator import PredictionTypeEvaluator


class BinaryClassificationEvaluator(PredictionTypeEvaluator):
    def prediction_formatter(self, predictions: np.ndarray) -> np.ndarray:
        return np.round(predictions)

    def evaluate(self, predictions: np.ndarray, targets: np.ndarray) -> Dict:
        result = BinaryClassificationEvaluator.bac(predictions=predictions, targets=targets)
        result['mse'] = self.mse(predictions=predictions[:, 0], targets=targets)
        return result

    @staticmethod
    def cf_matrix(predictions: np.ndarray, targets: np.ndarray) -> Dict:
        y_pred = np.round(predictions)[:, 0]
        cf = confusion_matrix(y_true=targets, y_pred=y_pred)
        print(cf)
        tn, fp, fn, tp = cf.ravel()
        n = tn + fp + fn + tp
        stats: Dict = {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        }

        for k, v in stats.copy().items():
            stats[f"{k} (%)"] = round(v / n * 100, 2)

        stats['tpr (%)'] = round(tp / (tp + fn) * 100, 2)
        stats['tnr (%)'] = round(tn / (tn + fp) * 100, 2)

        return stats

    @staticmethod
    def bac(predictions: np.ndarray, targets: np.ndarray) -> Dict:
        cf = BinaryClassificationEvaluator.cf_matrix(predictions=predictions, targets=targets)
        result = cf.copy()
        result['bac (%)'] = round((cf['tpr (%)'] + cf['tnr (%)']) / 2, 2)
        return result
