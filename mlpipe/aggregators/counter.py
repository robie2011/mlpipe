import numpy as np

from mlpipe.aggregators.abstract_numpy_reduction import AbstractNumpyReduction


class Counter(AbstractNumpyReduction):
    def __init__(self):
        super().__init__(reduce_func=np.ma.count)

    def javascript_group_aggregation(self):
        return "(a,b) => a+b"
