import logging
from abc import ABC, abstractmethod

from mlpipe.integration import PredictionResult
from mlpipe.mixins.logger_mixin import InstanceLoggerMixin

module_logger = logging.getLogger(__name__)


class AbstractOutput(ABC, InstanceLoggerMixin):
    @abstractmethod
    def write(self, result: PredictionResult):
        pass
