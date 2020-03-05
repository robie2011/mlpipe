import itertools
from typing import Tuple

import numpy as np

from mlpipe.processors.interfaces import AbstractProcessor
from mlpipe.processors.standard_data_format import StandardDataFormat


def _get_range_exceeding_maximum_length(ix_start: np.ndarray, ix_end: np.ndarray, threshold: int):
    size = ix_end - ix_start
    indexes = np.arange(size.size)
    out = [np.arange(
        ix_start[i] + threshold,
        ix_end[i] + 1).tolist() for i in indexes[size > threshold - 1]]

    return out


def _flatten_list(data):
    return list(itertools.chain(*data))


def _get_mask(data: np.ndarray, threshold: int):

    """
    Algorithm is explained by exampel below:

        Given is 1D-Variable called value.
        We create an array called next_equal which gives TRUE if next value is equal.
        We create an array called freeze_change which represents points on which
            freezed value has started or stopped.
        By getting only TRUE values from freeze_change-array we have all change points (array ix_change_col).
        Every 2nd point of ix_change_col represents starting point of freezed value.
        Every 2nd point starting from index=1 represents end point of freezd value.
        Now we can calculate difference between start and endpoint and handle difference more than threshold.


    Note:
        - freez_change on table below needs offset: -1
        - (d) is a dummy value on top and bottom of array

    | Index | Value | next_equal | freeze_change |
    |-------|-------|------------|---------------|
    |     0 |    20 | FALSE (d)  |               |
    |     1 |    21 | FALSE      | FALSE         |
    |     2 |    22 | FALSE      | FALSE         |
    |     3 |    20 | FALSE      | FALSE         |
    |     4 |    20 | TRUE       | TRUE          |
    |     5 |    20 | TRUE       | FALSE         |
    |     6 |    20 | TRUE       | FALSE         |
    |     7 |    20 | TRUE       | FALSE         |
    |     8 |    22 | FALSE      | TRUE          |
    |     9 |    23 | FALSE      | FALSE         |
    |       |       | FALSE (d)  | FALSE         |


    """

    n_rows, n_cols = data.shape
    data_mask = np.zeros(data.shape, dtype='bool')

    # next_equal: boolean shows whether following measurement has same value
    # dummy data on top and bottom of match-array for counting freed values
    dummy_data = [[False] * n_cols]
    next_equal = np.r_[
        dummy_data,
        data[1:] == data[:-1],
        dummy_data]

    freeze_change = next_equal[1:] != next_equal[:-1]
    indexes = np.arange(freeze_change.shape[0])

    for col in range(n_cols):
        ix_change_col = indexes[freeze_change[:, col]]
        ix_start = ix_change_col[::2]
        ix_end = ix_change_col[1::2]
        ranges = _get_range_exceeding_maximum_length(ix_start, ix_end, threshold)
        data_mask[_flatten_list(ranges), col] = True

    return data_mask


class FreezedValueRemover(AbstractProcessor):
    def __init__(self, max_freezed_values: int):
        self.max_freezed_values = max_freezed_values

    def _process2d(self, processor_input: StandardDataFormat) -> StandardDataFormat:
        mask = _get_mask(processor_input.data, self.max_freezed_values)
        data = processor_input.data.copy()
        data[mask] = np.nan
        return processor_input.modify_copy(data=data)
