from typing import List

import numpy as np

from mlpipe.dsl_interpreter.descriptions import InputOutputField
from .abstract_numpy_reduction import AbstractNumpyReduction


class Min(AbstractNumpyReduction):
    def __init__(self, sequence: int, generate: List[InputOutputField] = ()):
        super().__init__(reduce_func=np.min, generate=generate, sequence=sequence)

    def javascript_group_aggregation(self):
        return "(a,b) => Math.min(a,b)"
