from .abstract_processor import AbstractProcessor
from datasources import DataResult


class UpperLimitProcessor(AbstractProcessor):
    def __init__(self, upper_limit: [float]):
        self.upper_limit = upper_limit

    def process(self, data: DataResult) -> DataResult:
        mask = ~(data.values > self.upper_limit).any(axis=1)

        return DataResult(
            values=data.values[mask],
            timestamps=data.timestamps[mask],
            columns=data.columns)
