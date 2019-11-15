import numpy as np

from aggregators.outlier import ColumnLimit
from processors import AbstractProcessor, ProcessorData
from aggregators import Outlier, AggregatorInput


class OutlierRemover(AbstractProcessor):
    def __init__(self, limits:[ColumnLimit]):
        self.limits = limits

    def process(self, processor_input: ProcessorData) -> ProcessorData:
        aggregator_input = AggregatorInput(grouped_data=np.ma.array(np.expand_dims(processor_input.data, axis=0)), raw_data=None)
        #affected_index = Outlier(self.limits).aggregate(aggregator_input).affected_index.sum(axis=2)
        affected_index = Outlier(self.limits).aggregate(aggregator_input).affected_index
        affected_index = np.squeeze(affected_index, axis=0)
        data = processor_input.data.copy()
        data[affected_index == True] = np.nan

        return ProcessorData(
            labels=processor_input.labels,
            data=data,
            timestamps=processor_input.timestamps
        )
