import numpy as np
from .abstract_numpy_reduction import AbstractNumpyReduction


class Percentile(AbstractNumpyReduction):
    def __init__(self, percentile: float, interpolation='linear'):
        if percentile > 100.0 or percentile < 0.0:
            raise ValueError("percentile must be a float between 0 and 100")

        super().__init__(reduce_func=np.percentile)
        self.kwargs['q'] = percentile
        self.kwargs['interpolation'] = interpolation
