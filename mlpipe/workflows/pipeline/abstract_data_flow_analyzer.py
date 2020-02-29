from abc import ABC, abstractmethod
from typing import Dict

from mlpipe.processors.standard_data_format import StandardDataFormat


class AbstractDataFlowAnalyzer(ABC):
    @abstractmethod
    def init_flow(self, description: Dict):
        pass

    @abstractmethod
    def before_pipe_handler(self, class_name: str, input_data: StandardDataFormat):
        pass

    @abstractmethod
    def after_pipe_handler(self, class_name: str, input_data: StandardDataFormat):
        pass


class NullDataFlowAnalyzer(AbstractDataFlowAnalyzer):
    def init_flow(self, description: Dict):
        pass

    def before_pipe_handler(self, class_name: str, input_data: StandardDataFormat):
        pass

    def after_pipe_handler(self, class_name: str, input_data: StandardDataFormat):
        pass

