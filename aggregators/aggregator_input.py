import numpy as np
from typing import NamedTuple


class AggregatorInput(NamedTuple):
    grouped_data: np.ndarray


def create_aggregation_input(data: np.ndarray):
    data.flags.writable = False
    return AggregatorInput(data=data)

