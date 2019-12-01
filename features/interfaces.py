from abc import ABC, abstractmethod
from typing import NamedTuple
import numpy as np


class FeatureExtractor:
    @abstractmethod
    def extract(self, timestamps: np.ndarray, features: np.ndarray) -> np.ndarray:
        pass


class WindowedFeatureExtractorInput(NamedTuple):
    grouped_features: np.ndarray


class WindowedFeatureExtractor:
    @abstractmethod
    def extract(self, data: WindowedFeatureExtractorInput) -> np.ndarray:
        pass
