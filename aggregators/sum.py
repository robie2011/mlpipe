import numpy as np
from aggregators.abstract_numpy_reduction import AbstractNumpyReduction


class Sum(AbstractNumpyReduction):
    def __init__(self):
        super().__init__(reduce_func=np.sum)
