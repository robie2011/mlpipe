from dataclasses import dataclass
from typing import List, Dict

from mlpipe.utils.datautils import LabelSelector
from mlpipe.processors.column_selector import ColumnSelector
from mlpipe.processors.interfaces import AbstractProcessor
from mlpipe.processors.standard_data_format import StandardDataFormat
from mlpipe.workflows.utils import create_instance


@dataclass
class Scaler(AbstractProcessor):
    scaler: str
    fields: List[str]
    kwargs: Dict = frozenset()

    def _process2d(self, processor_input: StandardDataFormat) -> StandardDataFormat:
        partial_data = ColumnSelector(self.fields).process(processor_input).data
        transformer_restored = self.state
        if transformer_restored:
            partial_data = transformer_restored.transform(partial_data)
        else:
            transformer = create_instance(qualified_name=self.scaler, kwargs=self.kwargs)
            partial_data = transformer.fit_transform(partial_data)
            # note: state will be saved only during training process
            # each training has it unique identificator which is associated with state data
            self.state = transformer

        label_selection = LabelSelector(
            elements=processor_input.labels).select(self.fields)
        data = processor_input.data.copy()
        data[:, label_selection.indexes] = partial_data

        return processor_input.modify_copy(data=data)
