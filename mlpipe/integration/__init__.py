from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple

import numpy as np


@dataclass
class IntegrationResult:
    model_name: str
    session_id: str
    time_execution: datetime
    shape_initial: Tuple[int, int]
    shape_pipeline: Tuple[int, ...]
    timestamps: np.ndarray
    predictions: np.ndarray
