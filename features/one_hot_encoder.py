import numpy as np
from .interfaces import RawFeatureExtractor, RawFeatureExtractorInput
from sklearn import preprocessing as skpp


# TODO: Argument for encoding and transform should be 2D.
#       Otherwise exception will be thrown.
class OneHotEncoder(RawFeatureExtractor):
    def __init__(self, encoding: np.ndarray):
        self.encoding = encoding

    def extract(self, data: RawFeatureExtractorInput) -> np.ndarray:
        encoder = skpp.OneHotEncoder()
        encoder.fit(self.encoding)
        return encoder.transform(data.features).toarray()
