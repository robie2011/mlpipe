from typing import List

import numpy as np

from mlpipe.aggregators.abstract_aggregator import AbstractAggregator
from mlpipe.dsl_interpreter.descriptions import InputOutputField
from .aggregator_output import AggregatorOutput


class Trend(AbstractAggregator):
    def __init__(self, sequence: int, generate: List[InputOutputField] = ()):
        super().__init__(generate=generate, sequence=sequence)

    # code only work during workflow 2 (training)
    # for analytics this (calculating trend) do not make any sense
    # because we can have different timeslices in one partition
    def aggregate(self, grouped_data: np.ndarray) -> AggregatorOutput:
        last_values = grouped_data[:, -1]
        first_values = grouped_data[:, 0]
        return last_values - first_values

    def javascript_group_aggregation(self):
        """
        missing information after aggregation
        """
        return ""
