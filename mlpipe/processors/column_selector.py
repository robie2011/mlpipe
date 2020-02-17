from typing import List
from .interfaces import AbstractProcessor
from .standard_data_format import StandardDataFormat
from ..datautils import LabelSelector


class ColumnSelector(AbstractProcessor):
    def __init__(self, columns: List[str], enable_regex=False):
        self._columns = columns
        self.enable_regex = enable_regex

    def process(self, processor_input: StandardDataFormat) -> StandardDataFormat:
        indexes = LabelSelector(elements=processor_input.labels).select(
            self._columns, enable_regex=self.enable_regex).indexes
        return self._select_columns(processor_input, indexes)

    @staticmethod
    def _select_columns(processor_input: StandardDataFormat, indexes: List[int]) -> StandardDataFormat:
        columns = indexes
        data = processor_input.data[:, columns]
        labels = [processor_input.labels[i] for i in columns]
        return StandardDataFormat(
            labels=labels,
            timestamps=processor_input.timestamps,
            data=data)
