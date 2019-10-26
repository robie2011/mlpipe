import numpy as np
from aggregators.abstract_aggregator import AbstractAggregator


class Max(AbstractAggregator):
    def aggregate(self, xxs: np.ndarray) -> np.ndarray:
        return np.max(xxs, axis=1)
