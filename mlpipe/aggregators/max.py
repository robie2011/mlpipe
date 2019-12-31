import numpy as np
from .abstract_numpy_reduction import AbstractNumpyReduction


class Max(AbstractNumpyReduction):
    def __init__(self):
        super().__init__(reduce_func=np.max)

