from abc import ABC, abstractmethod
import numpy as np
from .aggregator_output import AggregatorOutput


class AbstractAggregator(ABC):
    @abstractmethod
    def aggregate(self, grouped_data: np.ndarray) -> AggregatorOutput:

        """
        input: 3D-Numpy Array
        first axis represents date/time
        second axis represents time steps / group of values
        third axis represents different sensors

        output: aggregated values (2D)
        first axis represents datetime
        second axis represents different sensors
        """
        pass

    @abstractmethod
    def javascript_group_aggregation(self):
        """
        Will be used to aggregate multiple groups.
        If merging groups do not make sense (e.g. aggregate two percentile)
        return an empty string.
        Otherwise return a string which contains a javascript arrow-function that
        takes two numbers and return one number
        see: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Functions/Arrow_functions
        """
        pass
