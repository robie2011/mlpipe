from dataclasses import dataclass
from datetime import datetime
from typing import List

import numpy as np


@dataclass
class IntegrationResult:
    model_name: str
    session_id: str
    time_execution: datetime
    shape_initial: List[int]
    shape_pipeline: List[int]
    timestamps: np.ndarray
    predictions: np.ndarray