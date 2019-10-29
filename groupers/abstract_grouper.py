from abc import ABC, abstractmethod
from aggregators.aggregator_input import AggregatorInput
from aggregators.aggregator_output import AggregatorOutput
import numpy as np
from typing import NamedTuple


class GroupInput(NamedTuple):
    raw_data: np.ndarray
    timestamps: np.ndarray


class AbstractGrouper(ABC):
    @abstractmethod
    def group(self, data_input: GroupInput) -> np.ndarray:
        pass
