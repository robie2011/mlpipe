import numpy as np
from .abstract_grouper import AbstractGrouper
import pandas as pd


class HourGrouper(AbstractGrouper):
    def group(self, timestamps: np.ndarray, raw_data: np.ndarray) -> np.ndarray:
        return pd.Series(timestamps).dt.hour.values

    def get_pretty_group_names(self) -> [str]:
        return list(map(lambda i: f"{i}:00", range(24)))
