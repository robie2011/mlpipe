from typing import List
import numpy as np
from mlpipe.datasources.abstract_datasource_adapter import Field, AbstractDatasourceAdapter
from mlpipe.datasources.empa import EmpaCsvSourceAdapter
from mlpipe.processors.standard_data_format import StandardDataFormat

MODULE_INIT_TIME = np.datetime64('now')


class DemoLiveData(AbstractDatasourceAdapter):
    def __init__(self, pathToFile: str, fields: List[str], windowMinutes: int, reset_init_time=False):
        super().__init__(fields=fields)
        self.source_returns_alias = True

        global MODULE_INIT_TIME
        if reset_init_time:
            MODULE_INIT_TIME = np.datetime64('now')
        self.window_delta = np.timedelta64(windowMinutes, 'm')

        self.cached_data = EmpaCsvSourceAdapter(fields=fields, pathToFile=pathToFile).get()

        # we need to add a small duration for processing time till first query
        # otherwise first row will be skipped
        processing_time_fix = np.timedelta64(30, 's')
        init_time_diff = MODULE_INIT_TIME + processing_time_fix - self.cached_data.timestamps[0]

        # adjust timestamp: set first row to current time
        # and subtract window_delta to ensure we have enough row for first fetch
        self.cached_data.timestamps = self.cached_data.timestamps + init_time_diff - self.window_delta

    def _fetch(self, _fields: List[Field]) -> StandardDataFormat:
        date_end = np.datetime64('now')
        date_start = date_end - self.window_delta
        valid_ix = np.logical_and(
            self.cached_data.timestamps >= date_start,
            self.cached_data.timestamps < date_end)

        return self.cached_data.modify_copy(
            timestamps=self.cached_data.timestamps[valid_ix],
            data=self.cached_data.data[valid_ix, :]
        )
