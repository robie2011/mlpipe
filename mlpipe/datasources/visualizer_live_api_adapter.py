import datetime
from typing import List

from mlpipe.datasources.abstract_datasource_adapter import AbstractDatasourceAdapter, Field
from mlpipe.datasources.visualizer_api_adapter import VisualizerApiAdapter
from mlpipe.processors.standard_data_format import StandardDataFormat

EXTRA_MINUTES = 3


class VisualizerLiveApiAdapter(AbstractDatasourceAdapter):
    def __init__(self,
                 username: str,
                 password: str,
                 duration_minutes: int,
                 fields: List[str],
                 nrows: int = None,
                 print_output: bool = False):
        super().__init__(fields=fields)
        self.duration_minutes = duration_minutes
        self.nrows = nrows or duration_minutes
        self.print_output = print_output
        self.source = VisualizerApiAdapter(
            username=username,
            password=password,
            fields=fields,
            date_from=datetime.datetime.now().isoformat(),
            date_to=datetime.datetime.now().isoformat()
        )
        self.source_returns_alias = True

    def _setup_datetime(self):
        date_to = datetime.datetime.now()
        date_from = date_to - datetime.timedelta(minutes=self.duration_minutes + EXTRA_MINUTES)
        return date_from, date_to

    def _fetch(self, _fields: List[Field]) -> StandardDataFormat:
        self.source.date_from, self.source.date_to = self._setup_datetime()
        data = self.source.get()
        if self.print_output:
            print(data.to_dataframe())
        return data.modify_copy(
            data=data.data[-self.nrows:],
            timestamps=data.timestamps[-self.nrows:]
        )
