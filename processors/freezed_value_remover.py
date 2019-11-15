import numpy as np
from aggregators import AggregatorInput, FreezedValueCounter
from processors import AbstractProcessor, ProcessorData


class FreezedValueRemover(AbstractProcessor):
    def __init__(self, max_freezed_values: int):
        self.max_freezed_values = max_freezed_values

    def process(self, processor_input: ProcessorData) -> ProcessorData:
        aggregator_input = AggregatorInput(
            raw_data=None,
            grouped_data=np.ma.array(
                np.expand_dims(processor_input.data, axis=0)
            )
        )
        aggregator_result = FreezedValueCounter(
            max_freezed_values=self.max_freezed_values).aggregate(aggregator_input)

        affected_index = np.squeeze(aggregator_result.affected_index, axis=0)
        data = processor_input.data.copy()
        data[affected_index] = np.nan
        return ProcessorData(
            labels=processor_input.labels,
            timestamps=processor_input.timestamps,
            data=data)
