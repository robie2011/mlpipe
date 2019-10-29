import numpy as np
from groupers import GroupInput
from .abstract_grouper import AbstractGrouper
import pandas as pd


class HourGrouper(AbstractGrouper):
    def group(self, data_input: GroupInput) -> np.ndarray:
        hours = pd.Series(data_input.timestamps).dt.hour.values
        minutes = pd.Series(data_input.timestamps).dt.minute.values

        # For values exactly halfway between rounded decimal values,
        # NumPy rounds to the nearest even value.
        # Thus 1.5 and 2.5 round to 2.0, -0.5 and 0.5 round to 0.0, etc.
        return np.round((hours + minutes/60)).astype('int')

