from typing import List
import numpy as np

from mlpipe.processors.interfaces import AbstractProcessor
from mlpipe.processors.standard_data_format import StandardDataFormat
from mlpipe.utils.datautils import LabelSelector


class NanReplacer(AbstractProcessor):
    def __init__(self, fields: List[str], replacement: int = 0):
        self.replacement = replacement
        self.fields = fields

    def _process2d(self, processor_input: StandardDataFormat) -> StandardDataFormat:
        selector = LabelSelector(processor_input.labels).select(self.fields)
        data = processor_input.data.copy()
        for col_id in selector.indexes:
            selection = data[:, col_id]
            ix_nans = np.isnan(selection)
            data[ix_nans, col_id] = self.replacement

        return processor_input.modify_copy(data=data)

