import pandas as pd
from .interfaces import AbstractProcessor
# frequency:
#   - 'T' or 'min' for Minute
#   - read about valid frequency:
#     https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects
from .standard_data_format import StandardDataFormat


class Resampler(AbstractProcessor):
    def __init__(self, freq: str):
        self.freq = freq

    def _process2d(self, processor_input: StandardDataFormat) -> StandardDataFormat:
        df = pd.DataFrame(data=processor_input.data, index=processor_input.timestamps)
        data_resampled = df.resample(self.freq).asfreq()

        return StandardDataFormat(
            labels=processor_input.labels,
            timestamps=data_resampled.index.values,
            data=data_resampled.values
        )
