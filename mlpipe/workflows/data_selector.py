from dataclasses import dataclass
from typing import List

import numpy as np

from mlpipe.processors.column_selector import ColumnSelector
from mlpipe.processors.standard_data_format import StandardDataFormat


@dataclass
class ModelInputSet:
    x: np.ndarray


@dataclass
class ModelInputOutputSet(ModelInputSet):
    y: np.ndarray

    def to_tuple(self):
        return self.x, self.y


@dataclass
class ModelTrainTestSet(ModelInputOutputSet):
    test_ratio: float

    def _test_size(self):
        if self.test_ratio <= 0.0 or self.test_ratio >= 1.0:
            raise ValueError("Invalid ratio for test. Use value between 0 and 1.")
        return int(self.x.shape[0] * self.test_ratio)

    def get_train_set(self):
        n_test = self._test_size()
        ix_end = self.x.shape[0] - n_test
        print(ix_end)
        return ModelInputOutputSet(x =self.x[:ix_end], y=self.y[:ix_end])

    def get_test_set(self):
        n_test = self._test_size()
        ix_start = self.x.shape[0] - n_test
        return ModelInputOutputSet(x =self.x[ix_start:], y=self.y[ix_start:])

    @staticmethod
    def from_model_input_output(data: ModelInputOutputSet, test_ratio: float):
        return ModelTrainTestSet(x=data.x, y=data.y, test_ratio=test_ratio)


def convert_to_model_input_set(input_data: StandardDataFormat, input_labels: List[str]):
    x_set = ColumnSelector(columns=input_labels, enable_regex=True).process(input_data)
    return ModelInputSet(x=x_set.data)


def convert_to_model_input_output_set(
        input_data: StandardDataFormat,
        input_labels: List[str],
        output_label: str):
    x_set = ColumnSelector(columns=input_labels, enable_regex=True).process(input_data)
    y_set = ColumnSelector(columns=[output_label], enable_regex=True).process(input_data)

    return ModelInputOutputSet(x=x_set.data, y=y_set.data.reshape(-1, ))
