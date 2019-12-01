import numpy as np
from .interfaces import FeatureExtractor, FeatureExtractorInput
from sklearn import preprocessing as skpp


# TODO: Argument for encoding and transform should be 2D.
#       Otherwise exception will be thrown.
class OneHotEncoder(FeatureExtractor):
    def __init__(self, encoding: np.ndarray):
        self.encoding = encoding

    def extract(self, timestamps: np.ndarray, features: np.ndarray) -> np.ndarray:
        encoder = skpp.OneHotEncoder()
        encoder.fit(self.encoding)
        return encoder.transform(features).toarray()
