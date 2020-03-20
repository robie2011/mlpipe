from typing import List

import numpy as np

from mlpipe.aggregators.abstract_aggregator import AbstractAggregator
from mlpipe.dsl_interpreter.descriptions import InputOutputField
from .aggregator_output import AggregatorOutput


class NanCounter(AbstractAggregator):
    def __init__(self, sequence: int, generate: List[InputOutputField] = ()):
        super().__init__(generate=generate, sequence=sequence)

    def aggregate(self, grouped_data: np.ndarray) -> AggregatorOutput:
        if not np.any(grouped_data.mask):
            print("1: no True / Mask value!")
        isnan = np.isnan(grouped_data)
        if isinstance(grouped_data, np.ma.MaskedArray):
            isnan[grouped_data.mask] = False

        return np.sum(isnan, axis=1)

    def javascript_group_aggregation(self):
        return "(a,b) => a + b"
