from typing import List
import numpy as np
from .interface import AbstractEncoder
from sklearn import preprocessing as skpp


# TODO: Argument for encoding and transform should be 2D.
#       Otherwise exception will be thrown.
class OneHotEncoder(AbstractEncoder):
    def __init__(self, encoding: List[int]):
        self.encoder = skpp.OneHotEncoder()
        self.encoder.fit(np.array(encoding).reshape(-1, 1))

    def encode(self, data_1d: np.ndarray) -> np.ndarray:
        return self.encoder.transform(data_1d.reshape(-1, 1)).toarray()
