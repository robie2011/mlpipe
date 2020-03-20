from typing import List

import numpy as np

from mlpipe.aggregators.abstract_aggregator import AbstractAggregator
from mlpipe.aggregators.aggregator_output import AggregatorOutput
from mlpipe.aggregators.counter import Counter
from mlpipe.dsl_interpreter.descriptions import InputOutputField


class MissingSamplesEstimation(AbstractAggregator):
    def __init__(self, sequence: int,
                 generate: List[InputOutputField] = ()):
        super().__init__(sequence=sequence, generate=generate)

    def aggregate(self, grouped_data: np.ndarray) -> AggregatorOutput:
        counter = Counter(sequence=self.sequence, generate=self.generate).aggregate(grouped_data)
        max_axis_1 = np.max(counter, axis=1)
        max_all_axis = np.max(max_axis_1, axis=0)
        return max_all_axis - counter

    def javascript_group_aggregation(self):
        return "(a,b) => a + b"
