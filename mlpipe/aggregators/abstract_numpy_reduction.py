from abc import abstractmethod
import numpy as np
from mlpipe.aggregators.abstract_aggregator import AbstractAggregator
from .aggregator_output import AggregatorOutput


class AbstractNumpyReduction(AbstractAggregator):

    @abstractmethod
    def javascript_group_aggregation(self):
        pass

    def __init__(self, reduce_func):
        self.reduce_func = reduce_func
        self.kwargs = {'axis': 1}

    def aggregate(self, grouped_data: np.ndarray) -> AggregatorOutput:
        return AggregatorOutput(metrics=self.reduce_func(grouped_data, **self.kwargs), affected_index=None)
