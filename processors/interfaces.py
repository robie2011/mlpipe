from typing import NamedTuple, Tuple, List
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class StandardDataFormat:
    timestamps: np.ndarray
    labels: List[str]
    data: np.ndarray


class AbstractProcessor(ABC):
    @abstractmethod
    def process(self, processor_input: StandardDataFormat) -> StandardDataFormat:
        pass
