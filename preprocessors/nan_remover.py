from .abstract_processor import AbstractProcessor
from datasources import DataResult
import numpy as np


class NanRemover(AbstractProcessor):
    def process(self, data: DataResult) -> DataResult:
        # 1. create boolean matrix which represents NaN as True
        # 2. reduce matrix "any(axis=1)" to 1d-array which
        #    represents True-value if any of columns has True value
        # 3. negate (~) array from step 2 and use it as mask for data
        mask = ~np.isnan(data.values).any(axis=1)

        return DataResult(
            values=data.values[mask],
            timestamps=data.timestamps[mask],
            columns=data.columns)
