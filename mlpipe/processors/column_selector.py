from typing import List
import numpy as np
from .interfaces import AbstractProcessor, StandardDataFormat
from ..datautils import LabelSelector


class ColumnSelector(AbstractProcessor):
    def __init__(self, columns: List[str]):
        self._columns = columns

    def process(self, processor_input: StandardDataFormat) -> StandardDataFormat:
        indexes = LabelSelector(elements=processor_input.labels).select(self._columns).indexes
        return self.select_columns(processor_input, indexes)

    @staticmethod
    def select_columns(processor_input: StandardDataFormat, indexes: List[int]) -> StandardDataFormat:
        columns = np.array(indexes)
        data = processor_input.data[:, columns]
        labels = [processor_input.labels[i] for i in columns]
        return StandardDataFormat(
            labels=labels,
            timestamps=processor_input.timestamps,
            data=data)
