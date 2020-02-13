from typing import List

import numpy as np

from mlpipe.datasources import EmpaCsvSourceAdapter
from mlpipe.processors.standard_data_format import StandardDataFormat

MODULE_INIT_TIME = np.datetime64('now')


class DemoLiveData(EmpaCsvSourceAdapter):
    def __init__(self, pathToFile: str, fields: List[str], windowMinutes: int, reset_init_time=False):
        super().__init__(pathToFile=pathToFile, fields=fields)
        self.cached_data = super()._fetch()
        if reset_init_time:
            self._reset_init_time()
        self.window_delta = np.timedelta64(windowMinutes, 'm')

        # we need to add a small duration for processing time till first query
        # otherwiese first row will be skipped
        processing_time_fix = np.timedelta64(30, 's')
        init_time_diff = MODULE_INIT_TIME + processing_time_fix - self.cached_data.timestamps[0]

        #
        # ajust timestamp: set first row to current time and subtract window_delta to ensure we have enough row for first fetch
        self.cached_data.timestamps = self.cached_data.timestamps + init_time_diff - self.window_delta

    def _reset_init_time(self):
        MODULE_INIT_TIME = np.datetime64('now')

    def _fetch(self) -> StandardDataFormat:
        date_end = np.datetime64('now')
        date_start = date_end - self.window_delta
        valid_ix = np.logical_and(self.cached_data.timestamps >= date_start, self.cached_data.timestamps < date_end)

        return self.cached_data.modify_copy(
            timestamps=self.cached_data.timestamps[valid_ix],
            data=self.cached_data.data[valid_ix, :]
        )