from dataclasses import dataclass
from typing import List

from sklearn.base import TransformerMixin

from mlpipe.datautils import LabelSelector
from mlpipe.processors import AbstractProcessor, StandardDataFormat
from mlpipe.processors.column_selector import ColumnSelector
from mlpipe.workflows.utils import create_instance


@dataclass
class Scaler(AbstractProcessor):
    name: str
    kwargs: object
    fields: List[str]
    saved_state: TransformerMixin = None

    def _transform(self, data):
        if self.saved_state:
            return self.saved_state.transform(data)
        else:
            self.saved_state = create_instance(qualified_name=self.name, kwargs=self.kwargs)
            return self.saved_state.fit_transform(data)

    def process(self, processor_input: StandardDataFormat) -> StandardDataFormat:
        partial_data = ColumnSelector(self.fields).process(processor_input).data
        partial_data = self._transform(partial_data)

        label_selection = LabelSelector(
            elements=processor_input.labels).select(self.fields)
        data = processor_input.data
        data[:, label_selection.indexes] = partial_data
        return processor_input.modify_copy(data=data)
