import numpy as np
from groupers import GroupInput
from .abstract_grouper import AbstractGrouper
import pandas as pd


class DayGrouper(AbstractGrouper):
    def group(self, data_input: GroupInput) -> np.ndarray:
        return pd.Series(data_input.timestamps).dt.day.values

