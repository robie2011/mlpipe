import numpy as np
import pandas as pd
from typing import Callable

from mlpipe.processors.interfaces import AbstractProcessor
from mlpipe.processors.standard_data_format import StandardDataFormat

_ITimeExtractor = Callable[[pd.Series], np.ndarray]

allowed_extractions: [(str, _ITimeExtractor)] = [
    # note: numpy round to nearest event value. Rounding 1.5 - 2.5 result in 2
    # see https://docs.scipy.org/doc/numpy/reference/generated/numpy.around.html#numpy.around
    ('hour', (lambda x:
              np.mod(np.round(x.dt.hour.values + x.dt.minute.values/60), 24)
              )),
    ('weekday', (lambda x: x.dt.weekday.values)),
    ('month', (lambda x: x.dt.month.values))
]


class TimeFeatureExtractor(AbstractProcessor):
    def __init__(self, extract: str, output_field: str):
        search = [func for name, func in allowed_extractions if name == extract]
        if len(search) == 0:
            raise Exception("extract parameter unknown: " + extract)

        self.output_field = output_field
        self._extractor: _ITimeExtractor = search[0]

    def process(self, processor_input: StandardDataFormat) -> StandardDataFormat:
        time_extracted = self._extractor(pd.Series(processor_input.timestamps)).reshape(-1, 1)
        data = np.hstack((processor_input.data, time_extracted))
        return StandardDataFormat(
            labels=processor_input.labels + [self.output_field],
            timestamps=processor_input.timestamps,
            data=data
        )
