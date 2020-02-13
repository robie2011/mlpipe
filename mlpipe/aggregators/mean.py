import numpy as np

from .abstract_numpy_reduction import AbstractNumpyReduction


class Mean(AbstractNumpyReduction):
    def __init__(self):
        super().__init__(reduce_func=np.mean)

    def javascript_group_aggregation(self):
        # We don't know number of elements of these groups.
        # This only works with groups of equal size
        return ""
