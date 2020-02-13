from dataclasses import dataclass
from typing import Optional, Dict

import numpy as np

from mlpipe.aggregators.abstract_aggregator import AbstractAggregator
from .aggregator_output import AggregatorOutput


@dataclass
class ColumnLimit(Dict):
    min: Optional[float]
    max: Optional[float]


# comparing nan throw warning: np.array([1, -1, np.nan]) > 0
# https://stackoverflow.com/questions/41130138/why-is-invalid-value-encountered-in-greater-warning-thrown-in-python-xarray-fo/41147570
# https://stackoverflow.com/questions/37651803/runtimewarning-invalid-value-encountered-in-greater
np.warnings.filterwarnings('ignore')


class Outlier(AbstractAggregator):
    def __init__(self, limits: [ColumnLimit]):
        self.limits = limits

    def aggregate(self, grouped_data: np.ndarray) -> AggregatorOutput:
        """note: use np.nan for no limit"""
        # we have a 1D array in which each element
        # define minimum for each sensor
        # e.g. minimums[0] define minimum for sensor = xxs[:, :, 0]
        min_matrix = np.zeros(grouped_data.shape, dtype='float')
        for i in range(len(self.limits)):
            min_matrix[:, :, i] = self.limits[i]['min'] if 'min' in self.limits[i] else np.nan
        min_filter = np.logical_and(
            np.invert(grouped_data.mask), grouped_data < min_matrix)

        max_matrix = np.zeros(grouped_data.shape, dtype='float')
        for i in range(len(self.limits)):
            max_matrix[:, :, i] = self.limits[i]['max'] if 'max' in self.limits[i] else np.nan
        max_filter = np.logical_and(
            np.invert(grouped_data.mask), grouped_data > max_matrix)

        affected_index = np.logical_or(min_filter, max_filter)
        return AggregatorOutput(metrics=np.sum(affected_index, axis=1), affected_index=affected_index)

    def javascript_group_aggregation(self):
        return "(a,b) => a + b"
