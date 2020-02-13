import numpy as np
from mlpipe.aggregators.abstract_aggregator import AbstractAggregator
from .aggregator_output import AggregatorOutput


class Trend(AbstractAggregator):
    def aggregate(self, grouped_data: np.ndarray) -> AggregatorOutput:
        last_values = grouped_data[:, -1]
        first_values = grouped_data[:, 0]
        return AggregatorOutput(metrics=last_values - first_values)

    def javascript_group_aggregation(self):
        """
        missing information after aggregation
        """
        return ""
