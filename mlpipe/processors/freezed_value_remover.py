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


def get_mask_for_freezed_values(data: np.ndarray, threshold: int):
    match = data[1:] == data[:-1]
    return _get_mask(data.shape, threshold=threshold, match=match)


def _get_mask(data_shape: Tuple[int, ...], threshold: int, match: np.ndarray):
    """
    Input: 2D matrix having different series on each column.
    detecting unchanged values for more than `threshold`-times
    return: 2D matrix with boolean value representing invalid (TRUE) values
    """

    n_rows, n_cols = data_shape
    data_mask = np.zeros(data_shape, dtype='bool')

    dummy_data = [[False] * n_cols]
    mask = np.r_[dummy_data, match, dummy_data]
    has_diff_to_previous: np.ndarray = mask[1:] != mask[:-1]
    indexes = np.arange(has_diff_to_previous.shape[0])

    for col in range(n_cols):
        ix_change_col = indexes[has_diff_to_previous[:, col]]
        ix_start = ix_change_col[::2]
        ix_end = ix_change_col[1::2]
        ranges = _get_range_exceeding_maximum_length(ix_start, ix_end, threshold)
        data_mask[_flatten_list(ranges), col] = True

    return data_mask


class FreezedValueRemover(AbstractProcessor):
    def __init__(self, max_freezed_values: int):
        self.max_freezed_values = max_freezed_values

    def _process2d(self, processor_input: StandardDataFormat) -> StandardDataFormat:
        mask = get_mask_for_freezed_values(processor_input.data, self.max_freezed_values)
        data = processor_input.data.copy()
        data[mask] = np.nan
        return processor_input.modify_copy(data=data)
