import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Tuple

import numpy as np

from mlpipe.utils.logger_mixin import InstanceLoggerMixin

module_logger = logging.getLogger(__name__)

@dataclass
class PredictionResult:
    model_name: str
    session_id: str
    time_execution: datetime
    shape_initial: Tuple[int, int]
    shape_pipeline: Tuple[int, ...]
    timestamps: np.ndarray
    predictions: np.ndarray


class AbstractOutput(ABC, InstanceLoggerMixin):
    @abstractmethod
    def write(self, result: PredictionResult):
        pass

