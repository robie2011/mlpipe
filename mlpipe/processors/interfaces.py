from abc import ABC, abstractmethod

from mlpipe.mixins.logger_mixin import InstanceLoggerMixin
from mlpipe.processors.standard_data_format import StandardDataFormat


class AbstractProcessor(ABC, InstanceLoggerMixin):
    state: object = None

    @abstractmethod
    def process(self, processor_input: StandardDataFormat) -> StandardDataFormat:
        pass

