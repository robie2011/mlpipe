import numpy as np
from aggregators.abstract_aggregator import AbstractAggregator
from aggregators.aggregator_input import AggregatorInput
from aggregators.aggregator_output import AggregatorOutput


class Percentile(AbstractAggregator):
    def __init__(self, percentile: float, interpolation='linear'):

        if percentile > 100.0 or percentile < 0.0:
            raise ValueError("percentile must be a float between 0 and 100")

        self.interpolation = interpolation
        self.percentile = percentile

    def aggregate(self, input_data: AggregatorInput) -> AggregatorOutput:
        return AggregatorOutput(
            metrics=np.percentile(input_data.grouped_data, q=self.percentile, axis=1, interpolation=self.interpolation))
