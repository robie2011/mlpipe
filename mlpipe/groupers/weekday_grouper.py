import numpy as np
import pandas as pd

from .abstract_grouper import AbstractGrouper


class WeekdayGrouper(AbstractGrouper):
    def group(self, timestamps: np.ndarray) -> np.ndarray:
        # note weekday start from monday. 0 = monday
        return pd.Series(timestamps.copy()).dt.weekday.values

    def get_pretty_group_names(self) -> [str]:
        return [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday"
        ]
