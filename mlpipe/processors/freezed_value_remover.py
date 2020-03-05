import numpy as np

from mlpipe.aggregators.freezed_value_counter import get_freezed_value_mask
from mlpipe.processors.interfaces import AbstractProcessor
from mlpipe.processors.standard_data_format import StandardDataFormat


class FreezedValueRemover(AbstractProcessor):
    def __init__(self, max_freezed_values: int):
        self.max_freezed_values = max_freezed_values

    def _process2d(self, processor_input: StandardDataFormat) -> StandardDataFormat:
        mask = get_freezed_value_mask(processor_input.data, self.max_freezed_values)
        data = processor_input.data.copy()
        data[mask] = np.nan
        return processor_input.modify_copy(data=data)
