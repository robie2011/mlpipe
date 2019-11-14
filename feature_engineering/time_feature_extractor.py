import numpy as np
from .interfaces import RawFeatureExtractor, RawFeatureExtractorInput
import pandas as pd
from typing import Callable
_ITimeExtractor = Callable[[pd.Series], np.ndarray]

allowed_extractions: [(str, _ITimeExtractor)] = [
    # note: numpy round to nearest event value. Rounding 1.5 - 2.5 result in 2
    # see https://docs.scipy.org/doc/numpy/reference/generated/numpy.around.html#numpy.around
    ('hour', (lambda x: np.round(x.dt.hour.values + x.dt.minute.values/60))),
    ('weekday', (lambda x: x.dt.weekday.values)),
    ('month', (lambda x: x.dt.month.values))
]


class TimeFeatureExtractor(RawFeatureExtractor):
    def __init__(self, extract: str):
        search = [func for name, func in allowed_extractions if name == extract]
        if len(search) == 0:
            raise Exception("extract parameter unknown: " + extract)

        self._extractor: _ITimeExtractor = search[0]

    def extract(self, data: RawFeatureExtractorInput) -> np.ndarray:
        return self._extractor(pd.Series(data.timestamps))
