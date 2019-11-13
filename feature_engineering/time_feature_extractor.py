import numpy as np
from .interfaces import RawFeatureExtractor, RawFeatureExtractorInput
import pandas as pd
from typing import Dict, Callable
_ITimeExtractor = Callable[[pd.Series], np.ndarray]

allowed_extractions: [(str, _ITimeExtractor)] = [
    ('hour', (lambda x: x.dt.hour.values)),
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
