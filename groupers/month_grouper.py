import numpy as np
from groupers import GroupInput
from .abstract_grouper import AbstractGrouper
import pandas as pd


class MonthGrouper(AbstractGrouper):
    def group(self, data_input: GroupInput) -> np.ndarray:
        return pd.Series(data_input.timestamps).dt.month.values

