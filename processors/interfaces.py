from typing import NamedTuple, Tuple, List
import numpy as np
from abc import ABC, abstractmethod


class ProcessorData(NamedTuple):
    labels: List[str]
    timestamps: np.ndarray
    data: np.ndarray


class AbstractProcessor(ABC):
    @abstractmethod
    def process(self, processor_input: ProcessorData) -> ProcessorData:
        pass
