from abc import ABC, abstractmethod
from typing import NamedTuple
import numpy as np


class RawFeatureExtractorInput(NamedTuple):
    timestamps: np.ndarray
    features: np.ndarray


class RawFeatureExtractor:
    @abstractmethod
    def extract(self, data: RawFeatureExtractorInput) -> np.ndarray:
        pass


class WindowedFeatureExtractorInput(NamedTuple):
    grouped_features: np.ndarray


class WindowedFeatureExtractor:
    @abstractmethod
    def extract(self, data: WindowedFeatureExtractorInput) -> np.ndarray:
        pass

