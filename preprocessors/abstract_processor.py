from abc import ABC, abstractmethod
import numpy as np
from datasources import DataResult


class AbstractProcessor(ABC):
    @abstractmethod
    def process(self, data: DataResult) -> DataResult:
        pass

