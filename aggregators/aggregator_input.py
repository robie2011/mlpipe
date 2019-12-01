import numpy as np
from typing import NamedTuple


class AggregatorInput(NamedTuple):
    grouped_data: np.ndarray


def create_aggregation_input(grouped_data: np.ndarray, raw_data: np.ndarray):
    grouped_data.flags.writable = False
    raw_data.flags.writable = False
    return AggregatorInput(grouped_data=grouped_data)

