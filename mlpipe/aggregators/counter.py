from mlpipe.aggregators import AbstractNumpyReduction
import numpy as np


class Counter(AbstractNumpyReduction):
    def __init__(self):
        super().__init__(reduce_func=np.ma.count)
