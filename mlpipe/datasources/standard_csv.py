from dataclasses import dataclass
from mlpipe.processors import StandardDataFormat
from .abstract_datasource_adapter import AbstractDatasourceAdapter
import pandas as pd
import os


@dataclass
class StandardCsvSourceAdapter(AbstractDatasourceAdapter):
    pathToFile: str
    sep=','


    def test(self):
        if os.path.isfile(self.pathToFile):
            return True
        else:
            return "File not found " + self.pathToFile

    def fetch(self) -> StandardDataFormat:
        data = pd.read_csv(
            self.pathToFile,
            sep=self.sep,
            parse_dates=['timestamp'])

        timestamps = data['timestamp'].values
        data = data.drop(labels='timestamp', axis=1)

        values = data.values
        return StandardDataFormat(
            labels=list(map(lambda x: x.strip(), data.columns.values.tolist())),
            data=values,
            timestamps=timestamps)
