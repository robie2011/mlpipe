from dataclasses import dataclass
from typing import List, cast
import numpy as np
from sklearn import preprocessing as skpp
from mlpipe.dsl_interpreter.descriptions import InputOutputField
from mlpipe.processors.column_dropper import ColumnDropper
from mlpipe.processors.column_selector import ColumnSelector
from mlpipe.processors.interfaces import AbstractProcessor
from mlpipe.processors.standard_data_format import StandardDataFormat


def _ensure_2d(data: np.ndarray):
    if len(data.shape) == 1:
        return data.reshape(-1, 1)
    else:
        return data


@dataclass
class RangeEncoder(AbstractProcessor):
    generate: List[InputOutputField]
    value_from: int
    value_to: int

    def _init_encoder(self) -> skpp.OneHotEncoder:
        if not self.state:
            self.state = cast(skpp.OneHotEncoder, self.state)  # just for ide

            self.state = skpp.OneHotEncoder()
            train_data = np.array(np.arange(self.value_from, self.value_to).tolist() * len(self.generate))
            train_data = train_data.reshape((-1, len(self.generate)))
            train_data = _ensure_2d(train_data)
            self.state.fit(train_data)

        return self.state

    def _process2d(self, processor_input: StandardDataFormat) -> StandardDataFormat:
        self.state = cast(skpp.OneHotEncoder, self.state)  # just for ide
        fields_input = [f['inputField'] for f in self.generate]
        fields_output = [f['outputField'] for f in self.generate]

        np_partial_data = ColumnSelector(columns=fields_input).process(processor_input=processor_input).data
        np_partial_data = _ensure_2d(np_partial_data)
        self._init_encoder()

        np_partial_data = self.state.transform(np_partial_data).toarray()
        partial_data_labels = []
        for ix, category in enumerate(self.state.categories_):
            for v in category.tolist():
                partial_data_labels.append(f"{fields_output[ix]}${v}")

        partial_data = StandardDataFormat(
            timestamps=processor_input.timestamps,
            data=np_partial_data,
            labels=partial_data_labels)

        merged_data = ColumnDropper(columns=fields_input).process(processor_input=processor_input)
        return merged_data.add_cols(partial_data)
