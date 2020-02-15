import os
from datetime import datetime
from typing import List

import pandas as pd

from .abstract_datasource_adapter import AbstractDatasourceAdapter, Field
from ..processors.standard_data_format import StandardDataFormat


class EmpaCsvSourceAdapter(AbstractDatasourceAdapter):
    def __init__(self, fields: List[str], pathToFile: str):
        self.pathToFile = pathToFile
        super().__init__(fields=fields)

    def _fetch(self, _fields: List[Field]) -> StandardDataFormat:
        data = pd.read_csv(
            self.pathToFile,
            sep=',',
            date_parser=lambda x: datetime.strptime(x, '%d.%m.%Y %H:%M:%S'),
            parse_dates=['_TIMESTAMP'])
        timestamps = data['_TIMESTAMP'].values

        field_names = set([f.name for f in _fields])
        names_todrop = set(data.columns.values) - field_names
        data = data.drop(labels=names_todrop, axis=1)

        values = data.values
        return StandardDataFormat(
            labels=data.columns.values.tolist(),
            data=values,
            timestamps=timestamps)
