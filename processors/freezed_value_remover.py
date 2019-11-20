import numpy as np
from processors import AbstractProcessor, ProcessorData
from .common_freezed_values import get_mask_for_freezed_values


class FreezedValueRemover(AbstractProcessor):
    def __init__(self, max_freezed_values: int):
        self.max_freezed_values = max_freezed_values

    def process(self, processor_input: ProcessorData) -> ProcessorData:
        mask = get_mask_for_freezed_values(processor_input.data, self.max_freezed_values)
        data = processor_input.data.copy()
        data[mask] = np.nan
        return ProcessorData(
            labels=processor_input.labels,
            timestamps=processor_input.timestamps,
            data=data)
