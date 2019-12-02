from abc import ABC, abstractmethod

import numpy as np


class AbstractGrouper(ABC):
    @abstractmethod
    def group(self, timestamps: np.ndarray, raw_data: np.ndarray) -> np.ndarray:
        pass

    def get_pretty_group_names(self) -> [str]:
        return []
