from dataclasses import dataclass
from random import random
from typing import List, cast
from mlpipe.datautils import LabelSelector
from mlpipe.encoders import AbstractEncoder
from mlpipe.processors import AbstractProcessor, StandardDataFormat, ColumnDropper
import numpy as np
import logging
from mlpipe.workflows.utils import create_instance

module_logger = logging.getLogger(__name__)


@dataclass
class Encoder(AbstractProcessor):
    name: str
    value_from: int
    value_to: int
    fields: List[str]
    kwargs: object = None

    def process(self, processor_input: StandardDataFormat) -> StandardDataFormat:
        label_selection = LabelSelector(processor_input.labels).select(self.fields)
        unchanged_data = ColumnDropper(self.fields).process(processor_input).data
        data_new = [unchanged_data]
        labels_new = []
        labels_new += label_selection.names_unselected

        for fieldname, ix_source_label in zip(label_selection.names, label_selection.indexes):
            data_selection = processor_input.data[:, ix_source_label]
            kwargs = self.kwargs or {}
            kwargs['value_from'] = self.value_from
            kwargs['value_to'] = self.value_to
            encoder = create_instance(qualified_name=self.name, kwargs=kwargs)
            encoder = cast(AbstractEncoder, encoder)
            qualified_classname = type(encoder).__name__
            module_logger.info("run encoding for field={0} with scaler={1}".format(
                fieldname, qualified_classname))

            data_encoded = encoder.encode(data_selection)
            data_new.append(data_encoded)
            encoding_id = "{0}_encoded$".format(fieldname, int(random() * 1000))
            labels_encoding = ["{0}_{1}".format(encoding_id, i) for i in range(data_encoded.shape[1])]
            labels_new += labels_encoding

        return processor_input.modify_copy(
            labels=labels_new,
            data=np.hstack(data_new),
        )
