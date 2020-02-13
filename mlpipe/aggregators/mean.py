from typing import List
import numpy as np
from .abstract_numpy_reduction import AbstractNumpyReduction
from ..dsl_interpreter.descriptions import InputOutputField


class Mean(AbstractNumpyReduction):
    def __init__(self, sequence: int, generate: List[InputOutputField] = ()):
        super().__init__(reduce_func=np.mean, generate=generate, sequence=sequence)

    def javascript_group_aggregation(self):
        # We don't know number of elements of these groups.
        # This only works with groups of equal size
        return ""
