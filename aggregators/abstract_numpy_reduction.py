from aggregators.abstract_aggregator import AbstractAggregator
from aggregators.aggregator_input import AggregatorInput
from aggregators.aggregator_output import AggregatorOutput


class AbstractNumpyReduction(AbstractAggregator):
    def __init__(self, reduce_func):
        self.reduce_func = reduce_func
        self.kwargs = {'axis': 1}

    def aggregate(self, input_data: AggregatorInput) -> AggregatorOutput:
        return AggregatorOutput(metrics=self.reduce_func(input_data.grouped_data, **self.kwargs), affected_index=None)
