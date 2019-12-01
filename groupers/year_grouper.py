import numpy as np
from .abstract_grouper import AbstractGrouper
import pandas as pd


class YearGrouper(AbstractGrouper):
    def group(self, timestamps: np.ndarray, raw_data: np.ndarray) -> np.ndarray:
        return pd.Series(timestamps).dt.year.values

