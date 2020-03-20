from typing import List

import numpy as np

from mlpipe.aggregators.abstract_aggregator import AbstractAggregator
from mlpipe.aggregators.aggregator_output import AggregatorOutput
from mlpipe.dsl_interpreter.descriptions import InputOutputField


class Counter(AbstractAggregator):
    def __init__(self, sequence: int, generate: List[InputOutputField] = ()):
        super().__init__(generate=generate, sequence=sequence)

    def aggregate(self, grouped_data: np.ndarray) -> AggregatorOutput:
        valid_values = np.invert(np.isnan(grouped_data))
        return np.sum(valid_values, axis=1)

    def javascript_group_aggregation(self):
        return "(a,b) => a+b"
