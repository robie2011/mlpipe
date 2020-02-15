import os
from dataclasses import dataclass
from typing import List

import pandas as pd

from .abstract_datasource_adapter import AbstractDatasourceAdapter, Field
from ..processors.standard_data_format import StandardDataFormat


@dataclass
class StandardCsvSourceAdapter(AbstractDatasourceAdapter):
    pathToFile: str
    sep = ','

    def _fetch(self, _fields: List[Field]) -> StandardDataFormat:
        data = pd.read_csv(
            self.pathToFile,
            sep=self.sep,
            parse_dates=['timestamp'])

        timestamps = data['timestamp'].values

        field_names = [f.name for f in _fields]
        names_todrop = ['_TIMESTAMP']
        names_todrop += [s for s in data.columns.values.tolist() if s not in field_names]
        data = data.drop(labels=names_todrop, axis=1)

        values = data.values
        return StandardDataFormat(
            labels=data.columns.values.tolist(),
            data=values,
            timestamps=timestamps
        )
