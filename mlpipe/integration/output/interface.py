import logging
from abc import ABC, abstractmethod

from mlpipe.integration import IntegrationResult
from mlpipe.mixins.logger_mixin import InstanceLoggerMixin
from mlpipe.workflows.utils import get_class_name

module_logger = logging.getLogger(__name__)


class AbstractOutput(ABC, InstanceLoggerMixin):
    @abstractmethod
    def write(self, result: IntegrationResult):
        pass
