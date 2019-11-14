from typing import NamedTuple
import numpy as np


class AggregatorOutput(NamedTuple):
    metrics: np.ndarray
    affected_index: np.ndarray  # 3D boolean array
