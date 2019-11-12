import numpy as np
from aggregators.abstract_aggregator import AbstractAggregator
from aggregators.aggregator_input import AggregatorInput
from aggregators.aggregator_output import AggregatorOutput


class Trend(AbstractAggregator):
    def aggregate(self, input_data: AggregatorInput) -> AggregatorOutput:
        last_values = input_data.grouped_data[:,-1]
        first_values = input_data.grouped_data[:,0]
        return AggregatorOutput(metrics=last_values - first_values)
