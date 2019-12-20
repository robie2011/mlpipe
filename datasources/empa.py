import pickle
from config import dirs
from processors import StandardDataFormat
from .abstract_datasource_adapter import AbstractDatasourceAdapter
import pandas as pd
from datetime import datetime
import os
import hashlib

ENABLE_CACHING = True

class EmpaCsvSourceAdapter(AbstractDatasourceAdapter):
    def __init__(self, pathToFile: str):
        self.pathToFile = pathToFile
        super().__init__()

    def test(self):
        if os.path.isfile(self.pathToFile):
            return True
        else:
            return "File not found " + self.pathToFile

    def fetch(self) -> StandardDataFormat:
        return self.fetch_cache_or_data()

    def fetch_data(self) -> StandardDataFormat:
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

    def fetch_cache_or_data(self):
        cache_id = hashlib.sha256(self.pathToFile.encode("utf-8")).hexdigest()
        path_to_cache = os.path.join(dirs.tmp, "cache_{0}".format(cache_id))
        if os.path.isfile(path_to_cache):
            with open(path_to_cache, "rb") as f:
                return pickle.load(f)
        else:
            data = self.fetch_data()
            with open(path_to_cache, "wb") as f:
                pickle.dump(data, f)
                return data
