import numpy as np
import pandas as pd

from .abstract_grouper import AbstractGrouper


class YearGrouper(AbstractGrouper):
    def group(self, timestamps: np.ndarray, raw_data: np.ndarray) -> np.ndarray:
        return pd.Series(timestamps).dt.year.values
