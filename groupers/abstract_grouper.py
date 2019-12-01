from abc import ABC, abstractmethod
from aggregators.aggregator_input import AggregatorInput
from aggregators.aggregator_output import AggregatorOutput
import numpy as np
from typing import NamedTuple


class AbstractGrouper(ABC):
    @abstractmethod
    def group(self, timestamps: np.ndarray, raw_data: np.ndarray) -> np.ndarray:
        pass

    def get_pretty_group_names(self) -> [str]:
        return []
