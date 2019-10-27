import numpy as np
from aggregators.abstract_aggregator import AbstractAggregator
from aggregators.aggregator_input import AggregatorInput
from aggregators.aggregator_output import AggregatorOutput

# comparing nan throw warning: np.array([1, -1, np.nan]) > 0
# https://stackoverflow.com/questions/41130138/why-is-invalid-value-encountered-in-greater-warning-thrown-in-python-xarray-fo/41147570
# https://stackoverflow.com/questions/37651803/runtimewarning-invalid-value-encountered-in-greater
np.warnings.filterwarnings('ignore')


class Outlier(AbstractAggregator):
    def __init__(self, limits: [{}]):
        self.limits = limits

    def aggregate(self, input_data: AggregatorInput) -> AggregatorOutput:
        """note: use np.nan for no limit"""
        # we have a 1D array in which each element
        # define minimum for each sensor
        # e.g. minimums[0] define minimum for sensor = xxs[:, :, 0]
        min_matrix = np.zeros(input_data.data.shape, dtype='float')
        for i in range(len(self.limits)):
            min_matrix[:, :, i] = self.limits[i]['min']
        min_filter = np.logical_and(np.invert(np.isnan(min_matrix)), input_data.data < min_matrix)
        min_count = np.sum(min_filter, axis=1)

        max_matrix = np.zeros(input_data.data.shape, dtype='float')
        for i in range(len(self.limits)):
            max_matrix[:, :, i] = self.limits[i]['max']
        max_count = np.sum(np.logical_and(np.invert(np.isnan(max_matrix)), input_data.data > max_matrix), axis=1)

        return AggregatorOutput(metrics=min_count + max_count)
