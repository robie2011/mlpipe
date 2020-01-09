from mlpipe.datasources import EmpaCsvSourceAdapter
from mlpipe.processors import StandardDataFormat
import numpy as np


class DemoLiveData(EmpaCsvSourceAdapter):
    def __init__(self, pathToFile: str, window_minutes: int):
        super().__init__(pathToFile)
        self.cached_data = super().fetch()
        self.window_delta = np.timedelta64(window_minutes, 'm')
        init_time_diff = np.datetime64('now') - self.cached_data.timestamps[0]

        #
        # ajust timestamp: set first row to current time and subtract window_delta to ensure we have enough row for first fetch
        self.cached_data.timestamps = self.cached_data.timestamps + init_time_diff - self.window_delta

    def fetch(self) -> StandardDataFormat:
        date_end = np.datetime64('now')
        date_start = date_end - self.window_delta
        valid_ix = np.logical_and(self.cached_data.timestamps >= date_start, self.cached_data.timestamps < date_end)

        return self.cached_data.modify_copy(
            timestamps=self.cached_data.timestamps[valid_ix],
            data=self.cached_data.data[valid_ix, :]
        )