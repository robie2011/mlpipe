from .abstract_preprocessor import AbstractPreprocessor
from datasources import DataResult


class UpperLimitProcessor(AbstractPreprocessor):
    def __init__(self, upper_limit: [float]):
        self.upper_limit = upper_limit

    def process(self, data: DataResult) -> DataResult:
        mask = ~(data.values > self.upper_limit).any(axis=1)

        return DataResult(
            values=data.values[mask],
            timestamps=data.timestamps[mask],
            columns=data.columns)
