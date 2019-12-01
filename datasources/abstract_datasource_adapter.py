from abc import ABC, abstractmethod
import numpy as np
from typing import NamedTuple, List


class AggregatorInput(NamedTuple):
    grouped_data: np.ndarray
    raw_data: np.ndarray


class DataResult(NamedTuple):
    values: np.ndarray
    timestamps: np.ndarray
    columns: List[str]


class AbstractDatasourceAdapter(ABC):
    @abstractmethod
    def test(self):
        pass

    @abstractmethod
    def fetch(self) -> DataResult:
        pass

