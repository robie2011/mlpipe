import numpy as np
from groupers import GroupInput
from .abstract_grouper import AbstractGrouper
import pandas as pd


class HourGrouper(AbstractGrouper):
    def group(self, data_input: GroupInput) -> np.ndarray:
        return pd.Series(data_input.timestamps).dt.hour.values

    def get_pretty_group_names(self) -> [str]:
        return list(map(lambda i: f"{i}:00", range(24)))
