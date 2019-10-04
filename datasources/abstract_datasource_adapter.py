from abc import ABC, abstractmethod
from .datasource import Datasource
import numpy as np


class DataResult:
    def __init__(self, values: np.ndarray, timestamps: np.ndarray, columns: [str]):
        self.values = values
        self.timestamps = timestamps
        self.columns = columns


class AbstractDatasourceAdapter(ABC):
    def __init__(self, *args):
        super(AbstractDatasourceAdapter, self).__init__(*args)

    @abstractmethod
    def test(self, source: Datasource):
        pass

    @abstractmethod
    def fetch(self, source: Datasource) -> DataResult:
        pass

