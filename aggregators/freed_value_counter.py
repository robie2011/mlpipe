import numpy as np
from aggregators.abstract_aggregator import AbstractAggregator
from aggregators.aggregator_input import AggregatorInput
from aggregators.aggregator_output import AggregatorOutput


class FreezedValueCounter(AbstractAggregator):
    def __init__(self, max_freezed_values: int):
        if (max_freezed_values < 0):
            raise ValueError(f"max_freezed_values must be greather or equal 0")
        self.max_freezed_values = max_freezed_values

    def aggregate(self, input_data: AggregatorInput) -> AggregatorOutput:
        # step 1: filter indexes which has same value at position i and on position i+max_freezed_values
        # step 2: all indexes which has 3 o
        return AggregatorOutput(metrics=np.max(input_data.grouped_data, axis=1))
