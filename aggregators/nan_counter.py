import numpy as np
from aggregators.abstract_aggregator import AbstractAggregator
from aggregators.aggregator_input import AggregatorInput
from aggregators.aggregator_output import AggregatorOutput


class NanCounter(AbstractAggregator):
    def aggregate(self, input_data: AggregatorInput) -> AggregatorOutput:
        nan_values = np.isnan(input_data.data)
        return AggregatorOutput(metrics=np.add.reduce(nan_values, axis=1))
