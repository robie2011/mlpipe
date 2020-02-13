from typing import List
import numpy as np
from .abstract_numpy_reduction import AbstractNumpyReduction
from ..dsl_interpreter.descriptions import InputOutputField


class Min(AbstractNumpyReduction):
    def __init__(self, sequence: int, generate: List[InputOutputField] = ()):
        super().__init__(reduce_func=np.min, generate=generate, sequence=sequence)

    def javascript_group_aggregation(self):
        return "(a,b) => Math.min(a,b)"
