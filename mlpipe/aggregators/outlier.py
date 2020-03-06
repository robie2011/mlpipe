from typing import Optional, List

import numpy as np

from mlpipe.aggregators.abstract_aggregator import AbstractAggregator
from mlpipe.dsl_interpreter.descriptions import InputOutputField
from .aggregator_output import AggregatorOutput

# comparing nan throw warning: np.array([1, -1, np.nan]) > 0
# https://stackoverflow.com/questions/41130138/why-is-invalid-value-encountered-in-greater-warning-thrown-in-python-xarray-fo/41147570
# https://stackoverflow.com/questions/37651803/runtimewarning-invalid-value-encountered-in-greater
np.warnings.filterwarnings('ignore')


class InputOutputLimits(InputOutputField):
    min: Optional[int]
    max: Optional[int]


def _is_ma(data: np.ndarray):
    return isinstance(data, np.ma.MaskedArray)


class Outlier(AbstractAggregator):
    # note: without InputOutputLimits this class is useless. No defaults possible.
    def __init__(self, sequence: int, generate: List[InputOutputLimits]):
        super().__init__(generate=generate, sequence=sequence)

    def aggregate(self, grouped_data: np.ndarray) -> AggregatorOutput:
        affected_index = self.affected_index(grouped_data)
        return np.sum(affected_index, axis=1)

    def affected_index(self, grouped_data: np.ndarray) -> np.ndarray:
        """note: use np.nan for no limit"""
        # we have a 1D array in which each element
        # define minimum for each sensor
        # e.g. minimums[0] define minimum for sensor = xxs[:, :, 0]
        min_matrix = np.zeros(grouped_data.shape, dtype='float')
        max_matrix = np.zeros(grouped_data.shape, dtype='float')

        for i, config in enumerate(self.generate):
            min_matrix[:, :, i] = config.get('min', np.nan)
            max_matrix[:, :, i] = config.get('max', np.nan)

        mask_filter = True
        if _is_ma(grouped_data):
            mask_filter = np.invert(grouped_data.mask)

        min_filter = np.logical_and(mask_filter, grouped_data < min_matrix)
        max_filter = np.logical_and(mask_filter, grouped_data > max_matrix)

        return np.logical_or(min_filter, max_filter)

    def javascript_group_aggregation(self):
        return "(a,b) => a + b"
