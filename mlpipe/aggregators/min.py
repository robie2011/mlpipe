import numpy as np
from .abstract_numpy_reduction import AbstractNumpyReduction


class Min(AbstractNumpyReduction):
    def __init__(self):
        super().__init__(reduce_func=np.min)

    def javascript_group_aggregation(self):
        return "(a,b) => Math.min(a,b)"
