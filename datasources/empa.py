from .abstract_datasource_adapter import AbstractDatasourceAdapter, DataResult
import pandas as pd
from datetime import datetime
import os


class EmpaCsvSourceAdapter(AbstractDatasourceAdapter):
    def __init__(self, pathToFile: str):
        self.pathToFile = pathToFile
        super().__init__()

    def test(self):
        if os.path.isfile(self.pathToFile):
            return True
        else:
            raise Exception("File not found")

    def fetch(self) -> DataResult:
        data = pd.read_csv(
            self.pathToFile,
            sep=',',
            date_parser=lambda x: datetime.strptime(x, '%d.%m.%Y %H:%M:%S'),
            parse_dates=['_TIMESTAMP'])
        timestamps = data['_TIMESTAMP'].values
        data = data.drop(labels='_TIMESTAMP', axis=1)
        values = data.values
        return DataResult(values=values, timestamps=timestamps, columns=data.columns)

