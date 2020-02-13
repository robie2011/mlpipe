from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class MultiAggregationResultCollector:
    data: np.ndarray
    labels: List[str]

    def hstack_bottom(self, labels: List[str], output: np.ndarray):
        rows_missing = self.data.shape[0] - output.shape[0]
        dummy_data = np.full((rows_missing, output.shape[1]), fill_value=np.nan, dtype='float')
        output = np.vstack((dummy_data, output))

        self.data = np.hstack((self.data, output))
        self.labels += labels
