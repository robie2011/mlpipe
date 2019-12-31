from abc import ABC, abstractmethod
import numpy as np


class AbstractEncoder(ABC):
    @abstractmethod
    def encode(self, data_1d: np.ndarray) -> np.ndarray:
        pass
