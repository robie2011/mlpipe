from abc import abstractmethod
from typing import List
import numpy as np
from mlpipe.aggregators.abstract_aggregator import AbstractAggregator
from .aggregator_output import AggregatorOutput
from mlpipe.dsl_interpreter.descriptions import InputOutputField


class AbstractNumpyReduction(AbstractAggregator):
    @abstractmethod
    def javascript_group_aggregation(self):
        pass

    def __init__(self, sequence: int, generate: List[InputOutputField], reduce_func):
        super().__init__(sequence=sequence, generate=generate)
        self.reduce_func = reduce_func
        self.kwargs = {'axis': 1}

    def aggregate(self, grouped_data: np.ndarray) -> AggregatorOutput:
        return self.reduce_func(grouped_data, **self.kwargs)
