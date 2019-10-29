import numpy as np
from groupers import GroupInput
from .abstract_grouper import AbstractGrouper
import pandas as pd


class YearGrouper(AbstractGrouper):
    def group(self, data_input: GroupInput) -> np.ndarray:
        return pd.Series(data_input.timestamps).dt.year.values

