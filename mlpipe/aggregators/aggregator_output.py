from typing import NamedTuple

import numpy as np


class AggregatorOutput(NamedTuple):
    metrics: np.ndarray

    # 3D boolean array
    affected_index: np.ndarray
