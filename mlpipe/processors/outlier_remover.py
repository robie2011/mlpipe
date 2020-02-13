import numpy as np

from mlpipe.aggregators import Outlier
from mlpipe.aggregators.outlier import ColumnLimit
from mlpipe.processors.interfaces import AbstractProcessor
from mlpipe.processors.standard_data_format import StandardDataFormat


class OutlierRemover(AbstractProcessor):
    def __init__(self, limits:[ColumnLimit]):
        self.limits = limits

    def process(self, processor_input: StandardDataFormat) -> StandardDataFormat:
        affected_index = Outlier(self.limits).aggregate(
            grouped_data=np.ma.array(np.expand_dims(processor_input.data, axis=0))).affected_index
        affected_index = np.squeeze(affected_index, axis=0)
        data = processor_input.data.copy()
        data[affected_index == True] = np.nan

        return StandardDataFormat(
            labels=processor_input.labels,
            data=data,
            timestamps=processor_input.timestamps
        )
