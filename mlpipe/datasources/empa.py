import os
from datetime import datetime
from typing import List

import pandas as pd

from .abstract_datasource_adapter import AbstractDatasourceAdapter
from ..processors.standard_data_format import StandardDataFormat


class EmpaCsvSourceAdapter(AbstractDatasourceAdapter):
    def __init__(self, fields: List[str], pathToFile: str):
        self.pathToFile = pathToFile
        super().__init__(fields=fields)

    def test(self):
        if os.path.isfile(self.pathToFile):
            return True
        else:
            return "File not found " + self.pathToFile

    def _fetch(self) -> StandardDataFormat:
        data = pd.read_csv(
            self.pathToFile,
            sep=',',
            date_parser=lambda x: datetime.strptime(x, '%d.%m.%Y %H:%M:%S'),
            parse_dates=['_TIMESTAMP'])
        timestamps = data['_TIMESTAMP'].values
        data = data.drop(labels='_TIMESTAMP', axis=1)
        values = data.values
        return StandardDataFormat(
            labels=list(map(lambda x: x.strip(), data.columns.values.tolist())),
            data=values,
            timestamps=timestamps)
