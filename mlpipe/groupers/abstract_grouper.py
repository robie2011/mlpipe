from abc import ABC, abstractmethod
from typing import List

import numpy as np


class AbstractGrouper(ABC):
    @abstractmethod
    def group(self, timestamps: np.ndarray) -> np.ndarray:
        pass

    def get_pretty_group_names(self) -> List[str]:
        return []
