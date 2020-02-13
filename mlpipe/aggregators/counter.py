from typing import List
import numpy as np
from mlpipe.aggregators.abstract_numpy_reduction import AbstractNumpyReduction
from mlpipe.dsl_interpreter.descriptions import InputOutputField


class Counter(AbstractNumpyReduction):
    def __init__(self, sequence: int, generate: List[InputOutputField] = ()):
        super().__init__(reduce_func=np.ma.count, generate=generate, sequence=sequence)

    def javascript_group_aggregation(self):
        return "(a,b) => a+b"
