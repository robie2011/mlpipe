from abc import ABC, abstractmethod
from .datasource import Datasource
import numpy as np
from typing import NamedTuple


class AggregatorInput(NamedTuple):
    grouped_data: np.ndarray
    raw_data: np.ndarray


class DataResult(NamedTuple):
    values: np.ndarray
    timestamps: np.ndarray
    columns: [str]


class AbstractDatasourceAdapter(ABC):
    def __init__(self, *args):
        super(AbstractDatasourceAdapter, self).__init__(*args)

    @abstractmethod
    def test(self, source: Datasource):
        pass

    @abstractmethod
    def fetch(self, source: Datasource) -> DataResult:
        pass

