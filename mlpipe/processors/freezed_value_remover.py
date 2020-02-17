import numpy as np

from mlpipe.processors.interfaces import AbstractProcessor
from mlpipe.processors.standard_data_format import StandardDataFormat
from .common_freezed_values import get_mask_for_freezed_values


class FreezedValueRemover(AbstractProcessor):
    def __init__(self, max_freezed_values: int):
        self.max_freezed_values = max_freezed_values

    def _process2d(self, processor_input: StandardDataFormat) -> StandardDataFormat:
        mask = get_mask_for_freezed_values(processor_input.data, self.max_freezed_values)
        data = processor_input.data.copy()
        data[mask] = np.nan
        return processor_input.modify_copy(data=data)
