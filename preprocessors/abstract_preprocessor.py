from abc import ABC, abstractmethod
import numpy as np
from datasources import DataResult


class AbstractPreprocessor(ABC):
    @abstractmethod
    def process(self, data: DataResult) -> DataResult:
        pass

