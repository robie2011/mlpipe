from abc import ABC, abstractmethod

from mlpipe.mixins.logger_mixin import InstanceLoggerMixin
from mlpipe.processors.standard_data_format import StandardDataFormat


class AbstractProcessor(ABC, InstanceLoggerMixin):
    state: object = None

    @abstractmethod
    def process(self, processor_input: StandardDataFormat) -> StandardDataFormat:
        pass

# todo:
#
# class AbstractValueModifierProcessor(AbstractProcessor):
#     fields: List[str]
#
#     def process(self, processor_input: StandardDataFormat) -> StandardDataFormat:
#         data = processor_input.data.copy()
#         if not self.fields:
#             data = self._modify(data)
#         else:
#             selection = LabelSelector(elements=processor_input.labels).select(selection=self.fields)
#             data[:, selection.indexes] = self._modify(data[:, selection.indexes])
#
#         return processor_input.modify_copy(data=data)
#
#     @abstractmethod
#     def _modify(self, data: np.ndarray) -> np.ndarray:
#         pass
#
