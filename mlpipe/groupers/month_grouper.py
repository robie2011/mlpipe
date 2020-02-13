import numpy as np
import pandas as pd

from .abstract_grouper import AbstractGrouper


class MonthGrouper(AbstractGrouper):
    def group(self, timestamps: np.ndarray, raw_data: np.ndarray) -> np.ndarray:
        return pd.Series(timestamps).dt.month.values

    def get_pretty_group_names(self) -> [str]:
        return [
            "N/A Zero Month do not exists",
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December"
        ]
