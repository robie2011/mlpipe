import pandas as pd

from mlpipe.processors.standard_data_format import StandardDataFormat
from .interfaces import AbstractProcessor


class Interpolation(AbstractProcessor):
    def __init__(self, max_consecutive_interpolated_value: int, method='linear'):
        self.threshold = max_consecutive_interpolated_value
        self.method = method

    def _process2d(self, processor_input: StandardDataFormat) -> StandardDataFormat:
        df = pd.DataFrame(data=processor_input.data)
        data = df.interpolate(method=self.method, limit_direction='forward', limit=self.threshold).values
        return StandardDataFormat(
            labels=processor_input.labels,
            timestamps=processor_input.timestamps,
            data=data
        )
