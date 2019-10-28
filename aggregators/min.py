import numpy as np
from aggregators.abstract_aggregator import AbstractAggregator
from aggregators.aggregator_input import AggregatorInput
from aggregators.aggregator_output import AggregatorOutput


class Min(AbstractAggregator):
    def aggregate(self, input_data: AggregatorInput) -> AggregatorOutput:
        return AggregatorOutput(metrics=np.min(input_data.grouped_data, axis=1))
