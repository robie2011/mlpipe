from typing import List
import numpy as np
from mlpipe.dsl_interpreter.descriptions import InputOutputField
from mlpipe.aggregators.abstract_aggregator import AbstractAggregator
from .aggregator_output import AggregatorOutput


class NanCounter(AbstractAggregator):
    def __init__(self, sequence: int, generate: List[InputOutputField] = ()):
        super().__init__(generate=generate, sequence=sequence)

    def aggregate(self, grouped_data: np.ndarray) -> AggregatorOutput:
        # todo:
        # use raw input and calculate where nan is found (index)
        # use intersection with indexes in group
        nan_values = np.isnan(grouped_data)

        # todo: this code is probably correct because we are using masked array. VERIFY.
        return np.add.reduce(nan_values, axis=1)

    def javascript_group_aggregation(self):
        return "(a,b) => a + b"
