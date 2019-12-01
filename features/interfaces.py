from abc import ABC, abstractmethod
import numpy as np


class FeatureExtractor:
    @abstractmethod
    def extract(self, timestamps: np.ndarray, features: np.ndarray) -> np.ndarray:
        pass


class WindowedFeatureExtractor:
    @abstractmethod
    def extract(self, grouped_features: np.ndarray) -> np.ndarray:
        pass


"""
multiple matrices
each row contains a sequence data or group data
"""
GroupedData = np.ndarray
