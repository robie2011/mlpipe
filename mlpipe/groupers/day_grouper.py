import numpy as np
import pandas as pd

from .abstract_grouper import AbstractGrouper


class DayGrouper(AbstractGrouper):
    def group(self, timestamps: np.ndarray) -> np.ndarray:
        return pd.Series(timestamps).dt.day.values
