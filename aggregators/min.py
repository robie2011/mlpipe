import numpy as np
from aggregators.AbstractNumpyReduction import AbstractNumpyReduction


class Min(AbstractNumpyReduction):
    def __init__(self):
        super().__init__(reduce_func=np.min)

