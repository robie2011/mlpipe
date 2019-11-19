import numpy as np
from aggregators import AggregatorInput, FreezedValueCounter
from processors import AbstractProcessor, ProcessorData
import itertools


def get_range_exceeding_maximum_length(ix_start: np.ndarray, ix_end: np.ndarray, threshold: int):
    size = ix_end - ix_start
    indexes = np.arange(size.size)
    out = [np.arange(
        ix_start[i] + threshold,
        ix_end[i] + 1).tolist() for i in indexes[size > threshold - 1]]

    return out


def flatten_list(data):
    return list(itertools.chain(*data))


def get_mask(data, threshold):
    """
    Input: 2D matrix having different series on each column.
    detecting unchanged values for more than `threshold`-times
    return: 2D matrix with boolean value representing invalid (TRUE) values
    """

    n_rows, n_cols = data.shape
    indexes = np.arange(n_rows)
    data_mask = np.zeros(data.shape, dtype='bool')

    dummy_data = [[False] * n_cols]
    mask = np.r_[dummy_data, data[1:] == data[:-1], dummy_data]
    has_diff_to_previous = mask[1:] != mask[:-1]

    for col in range(n_cols):
        ix_change_col = indexes[has_diff_to_previous[:, col]]
        ix_start = ix_change_col[::2]
        ix_end = ix_change_col[1::2]
        ranges = get_range_exceeding_maximum_length(ix_start, ix_end, threshold)
        data_mask[flatten_list(ranges), col] = True

    return data_mask


class FreezedValueRemover(AbstractProcessor):
    def __init__(self, max_freezed_values: int):
        self.max_freezed_values = max_freezed_values

    def process(self, processor_input: ProcessorData) -> ProcessorData:
        mask = get_mask(processor_input.data, self.max_freezed_values)
        data = processor_input.data.copy()
        data[mask] = np.nan
        return ProcessorData(
            labels=processor_input.labels,
            timestamps=processor_input.timestamps,
            data=data)
