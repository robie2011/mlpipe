import numpy as np
from groupers import GroupInput
from .abstract_grouper import AbstractGrouper
import pandas as pd


class WeekdayGrouper(AbstractGrouper):
    def group(self, data_input: GroupInput) -> np.ndarray:
        # note weekday start from monday. 0 = monday
        return pd.Series(data_input.timestamps).dt.weekday.values
