import numpy as np
from aggregators.abstract_aggregator import AbstractAggregator


class Min(AbstractAggregator):
    def aggregate(self, xxs: np.ndarray) -> np.ndarray:
        return np.min(xxs, axis=1)
