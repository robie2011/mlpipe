from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class StandardDataFormat:
    timestamps: np.ndarray
    labels: List[str]
    data: np.ndarray

    def modify_copy(self, labels: List[str] = None, timestamps: np.ndarray = None, data: np.ndarray = None):
        def get_or_default(a, default):
            if a is None:
                return default
            else:
                return a

        return StandardDataFormat(
            labels=get_or_default(labels, self.labels),
            timestamps=get_or_default(timestamps, self.timestamps),
            data=get_or_default(data, self.data)
        )


class AbstractProcessor(ABC):
    @abstractmethod
    def process(self, processor_input: StandardDataFormat) -> StandardDataFormat:
        pass
