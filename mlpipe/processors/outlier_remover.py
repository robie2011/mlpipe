from typing import List

import numpy as np

from mlpipe.aggregators.outlier import Outlier, InputOutputLimits
from mlpipe.datautils import LabelSelector
from mlpipe.dsl_interpreter.config_reader import ConfigReader
from mlpipe.processors.interfaces import AbstractProcessor
from mlpipe.processors.standard_data_format import StandardDataFormat


class OutlierRemover(AbstractProcessor):
    def __init__(self, generate: List[InputOutputLimits]):
        self.generate = generate

    def _process2d(self, processor_input: StandardDataFormat) -> StandardDataFormat:
        fields_in = [
            ConfigReader.from_dict(g).get_or_error(key="inputField", context="OutlierRemover Config")
            for g in self.generate]
        cols_selected = LabelSelector(elements=processor_input.labels).select(selection=fields_in).indexes

        # our group has only one member
        grouped_data = np.ma.array(np.expand_dims(processor_input.data[:, cols_selected], axis=0))

        affected_index = Outlier(sequence=np.nan, generate=self.generate).affected_index(grouped_data=grouped_data)
        affected_index = np.squeeze(affected_index, axis=0)

        data = processor_input.data.copy()
        t = data[:, cols_selected]
        t[affected_index] = np.nan
        data[:, cols_selected] = t

        return processor_input.modify_copy(data=data)
