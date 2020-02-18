from abc import ABC, abstractmethod
from mlpipe.mixins.logger_mixin import InstanceLoggerMixin
from mlpipe.processors.standard_data_format import StandardDataFormat
from mlpipe.workflows.utils import get_class_name


class AbstractProcessor(ABC, InstanceLoggerMixin):
    state: object = None
    id: str

    def process(self, processor_input: StandardDataFormat) -> StandardDataFormat:
        dim = len(processor_input.data.shape)
        if dim == 2:
            return self._process2d(processor_input)
        elif dim == 3:
            return self._process3d(processor_input)
        else:
            raise NotImplementedError(f"process operation not implemented for data dimension: {dim}")

    def _process3d(self, processor_input: StandardDataFormat) -> StandardDataFormat:
        raise NotImplementedError(
            f"3D-Data received for processing but class {get_class_name(self)} " +
            f"do not implement process operation for 3D-Data.")

    @abstractmethod
    def _process2d(self, processor_input: StandardDataFormat) -> StandardDataFormat:
        pass
