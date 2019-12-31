import numpy as np
from .interfaces import AbstractProcessor, StandardDataFormat
import pandas as pd


# frequency:
#   - 'T' or 'min' for Minute
#   - read about valid frequency:
#     https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects
class Resampler(AbstractProcessor):
    def __init__(self, freq: str):
        self._freq = freq

    def process(self, processor_input: StandardDataFormat) -> StandardDataFormat:
        df = pd.DataFrame(data=processor_input.data, index=processor_input.timestamps)
        data_resampled = df.resample(self._freq).asfreq()

        return StandardDataFormat(
            labels=processor_input.labels,
            timestamps=data_resampled.index.values,
            data=data_resampled.values
        )

