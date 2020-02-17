from typing import List
from .interfaces import AbstractProcessor
from .standard_data_format import StandardDataFormat
from ..datautils import LabelSelector


class ColumnSelector(AbstractProcessor):
    def __init__(self, columns: List[str], enable_regex=False):
        self._columns = columns
        self.enable_regex = enable_regex

    def _process2d(self, processor_input: StandardDataFormat) -> StandardDataFormat:
        indexes = LabelSelector(elements=processor_input.labels).select(
            self._columns, enable_regex=self.enable_regex).indexes
        return self._select_columns(processor_input, indexes)

    def _process3d(self, processor_input: StandardDataFormat) -> StandardDataFormat:
        indexes = LabelSelector(elements=processor_input.labels).select(
            self._columns, enable_regex=self.enable_regex).indexes
        return self._select_columns(processor_input, indexes, is_3d=True)

    @staticmethod
    def _select_columns(processor_input: StandardDataFormat, indexes: List[int], is_3d=False) -> StandardDataFormat:
        columns = indexes
        if is_3d:
            data = processor_input.data[:, :, columns]
        else:
            data = processor_input.data[:, columns]

        labels = [processor_input.labels[i] for i in columns]
        return StandardDataFormat(
            labels=labels,
            timestamps=processor_input.timestamps,
            data=data)
