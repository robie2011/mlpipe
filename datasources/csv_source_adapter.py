from .abstract_datasource_adapter import AbstractDatasourceAdapter, DataResult
from .datasource import Datasource
import os
import pandas as pd
from datetime import datetime


class CsvSourceAdapter(AbstractDatasourceAdapter):
    def __init__(self, *args):
        super(CsvSourceAdapter, self).__init__(*args)

    def test(self, source: Datasource):
        return os.path.isfile(source.connection_string)

    def fetch(self, source: Datasource) -> DataResult:
        return pd.read_csv(source.connection_string, sep=',')


class EmpaCsvSourceAdapter(CsvSourceAdapter):
    def __init__(self, *args):
        super(CsvSourceAdapter, self).__init__(*args)

    # todo: rewrite - use adapter specific config by passing it through constructor (kwargs)
    def fetch(self, source: Datasource) -> DataResult:
        data = pd.read_csv(
            source.connection_string, 
            sep=',',
            date_parser=lambda x: datetime.strptime(x, '%d.%m.%Y %H:%M:%S'),
            parse_dates=['_TIMESTAMP'])
        timestamps = data['_TIMESTAMP'].values
        data = data.drop(labels='_TIMESTAMP', axis=1)
        values = data.values
        return DataResult(values=values, timestamps=timestamps, columns=data.columns)

