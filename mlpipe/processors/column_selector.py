import numpy as np
from .interfaces import AbstractProcessor, StandardDataFormat


class ColumnSelector(AbstractProcessor):
    def __init__(self, columns: [str]):
        self._columns = columns

    def process(self, processor_input: StandardDataFormat) -> StandardDataFormat:
        columns = np.array([i for i, label
                            in enumerate(processor_input.labels)
                            if label in self._columns])

        return self.select_columns(processor_input, columns)

    @staticmethod
    def select_columns(processor_input: StandardDataFormat, columns: np.ndarray) -> StandardDataFormat:
        data = processor_input.data[:, columns]
        labels = [processor_input.labels[i] for i in columns]
        return StandardDataFormat(
            labels=labels,
            timestamps=processor_input.timestamps,
            data=data)
