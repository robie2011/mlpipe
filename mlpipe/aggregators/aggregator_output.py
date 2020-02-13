from typing import NamedTuple

import numpy as np


class AggregatorOutput(NamedTuple):
    metrics: np.ndarray
