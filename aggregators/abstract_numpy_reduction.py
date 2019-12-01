import numpy as np
from aggregators.abstract_aggregator import AbstractAggregator
from aggregators.aggregator_output import AggregatorOutput


class AbstractNumpyReduction(AbstractAggregator):
    def __init__(self, reduce_func):
        self.reduce_func = reduce_func
        self.kwargs = {'axis': 1}

    def aggregate(self, grouped_data: np.ndarray) -> AggregatorOutput:
        return AggregatorOutput(metrics=self.reduce_func(grouped_data, **self.kwargs), affected_index=None)
