from typing import List

import numpy as np

from mlpipe.dsl_interpreter.descriptions import InputOutputField
from .abstract_numpy_reduction import AbstractNumpyReduction


class Sum(AbstractNumpyReduction):
    def __init__(self, sequence: int, generate: List[InputOutputField] = ()):
        super().__init__(generate=generate, sequence=sequence, reduce_func=np.nansum)

    def javascript_group_aggregation(self):
        return "(a,b) => a + b"
