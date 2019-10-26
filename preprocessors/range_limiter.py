from .abstract_processor import AbstractProcessor
from datasources import DataResult
import numpy as np

# column: 1025
# max: 31
# min: null
# action: "replace"
# replace_value: 10


class RangeLimiter(AbstractProcessor):
    def __init__(self, limits: [{}]):
        self.limits = limits

    def _get_minimums(self):
        return np.fromiter(map(lambda x: x['min'], self.limits), dtype='float')

    def _get_maximums(self):
        return np.fromiter(map(lambda x: x['max'], self.limits), dtype='float')

    def process(self, data: DataResult) -> DataResult:
        # todo
        mask = ~(data.values > self.limits).any(axis=1)

        return DataResult(
            values=data.values[mask],
            timestamps=data.timestamps[mask],
            columns=data.columns)
