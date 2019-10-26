import numpy as np
from aggregators.abstract_aggregator import AbstractAggregator


class Mean(AbstractAggregator):
    def aggregate(self, xxs: np.ndarray) -> object:
        return np.mean(xxs, axis=1)
