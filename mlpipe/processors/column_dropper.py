from typing import List
import numpy as np
from .column_selector import ColumnSelector
from .interfaces import AbstractProcessor, StandardDataFormat


class ColumnDropper(AbstractProcessor):
    def __init__(self, columns: List[str]):
        self._columns = columns

    def process(self, processor_input: StandardDataFormat) -> StandardDataFormat:
        columns = np.array([i for i, label
                            in enumerate(processor_input.labels)
                            if label not in self._columns])
        return ColumnSelector.select_columns(processor_input, columns)
