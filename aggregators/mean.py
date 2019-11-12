import numpy as np
from aggregators.AbstractNumpyReduction import AbstractNumpyReduction


class Mean(AbstractNumpyReduction):
    def __init__(self):
        super().__init__(reduce_func=np.mean)
