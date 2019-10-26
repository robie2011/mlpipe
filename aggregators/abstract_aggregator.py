from abc import ABC, abstractmethod
import numpy as np


class AbstractAggregator(ABC):
    @abstractmethod
    def aggregate(self, xxs: np.ndarray) -> np.ndarray:
        """
        input: 3D-Numpy Array
        first axis represents date/time
        second axis represents time steps / group of values
        third axis represents different sensors

        output: aggregated values (2D)
        first axis represents datetime
        second axis represents different sensors
        """
        pass
