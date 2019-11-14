import numpy as np
from .interfaces import AbstractProcessor, ProcessorData


class ColumnDropper(AbstractProcessor):
    def __init__(self, columns: [str]):
        self._columns = columns

    def process(self, processor_input: ProcessorData) -> ProcessorData:
        columns = np.array([i for i, label
                            in enumerate(processor_input.labels)
                            if label not in self._columns])

        data = processor_input.data[:, columns]
        labels = [processor_input.labels[i] for i in columns]
        return ProcessorData(
            labels=labels,
            timestamps=processor_input.timestamps,
            data=data)
