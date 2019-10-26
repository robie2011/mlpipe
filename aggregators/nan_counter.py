import numpy as np
from aggregators.abstract_aggregator import AbstractAggregator


class NanCounter(AbstractAggregator):
    def aggregate(self, xxs: np.ndarray) -> np.ndarray:
        data = np.isnan(xxs)
        return np.add.reduce(data, axis=1)