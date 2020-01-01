import numpy as np
from .abstract_grouper import AbstractGrouper
import pandas as pd


class WeekdayGrouper(AbstractGrouper):
    def group(self, timestamps: np.ndarray, raw_data: np.ndarray) -> np.ndarray:
        # note weekday start from monday. 0 = monday
        return pd.Series(timestamps).dt.weekday.values

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