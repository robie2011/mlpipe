import numpy as np
import pandas as pd

from .abstract_grouper import AbstractGrouper


class HourGrouper(AbstractGrouper):
    def group(self, timestamps: np.ndarray) -> np.ndarray:
        return pd.Series(timestamps.copy()).dt.hour.values

    def get_pretty_group_names(self) -> [str]:
        return list(map(lambda i: f"{i}:00", range(24)))
