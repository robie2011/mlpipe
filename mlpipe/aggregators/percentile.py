from typing import List

import numpy as np

from .abstract_numpy_reduction import AbstractNumpyReduction
from ..dsl_interpreter.descriptions import InputOutputField


class Percentile(AbstractNumpyReduction):
    def __init__(self, sequence: int, percentile: float,
                 interpolation='linear',
                 generate: List[InputOutputField] = ()):
        super().__init__(generate=generate, sequence=sequence, reduce_func=np.nanpercentile)
        if percentile > 100.0 or percentile < 0.0:
            raise ValueError("percentile must be a float between 0 and 100")

        self.kwargs['q'] = percentile
        self.kwargs['interpolation'] = interpolation

    def javascript_group_aggregation(self):
        """
        we don't know distribution of these groups
        """
        return ""
