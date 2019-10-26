import numpy as np
from aggregators.abstract_aggregator import AbstractAggregator


class Percentile(AbstractAggregator):
    def __init__(self, percentile: float):

        if percentile > 100.0 or percentile < 0.0:
            raise ValueError("percentile must be a float between 0 and 100")

        self.percentile = percentile

    def aggregate(self, xxs: np.ndarray) -> np.ndarray:
        return np.percentile(xxs, seq=self.percentile, axis=1)
