import logging
from abc import ABC, abstractmethod

from mlpipe.integration import IntegrationResult
from mlpipe.workflows.utils import get_class_name

module_logger = logging.getLogger(__name__)


class AbstractOutput(ABC):
    def write(self, result: IntegrationResult):
        module_logger.info(f"using output class {get_class_name(self)}")
        self._write(result)

    @abstractmethod
    def _write(self, result: IntegrationResult):
        pass
