import numpy as np
import pandas as pd
from .interfaces import ProcessorData, AbstractProcessor


class LinearInterpolation(AbstractProcessor):
    def __init__(self, max_consecutive_interpolated_value: int):
        self.threshold = max_consecutive_interpolated_value

    def process(self, processor_input: ProcessorData) -> ProcessorData:
        df = pd.DataFrame(data=processor_input.data)
        data = df.interpolate(method='linear', limit_direction='forward', limit=self.threshold).values
        return ProcessorData(
            labels=processor_input.labels,
            timestamps=processor_input.timestamps,
            data=data
        )
