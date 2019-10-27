from abc import ABC, abstractmethod
#from aggregators import AggregatorOutput
from aggregators.aggregator_input import AggregatorInput
from aggregators.aggregator_output import AggregatorOutput


class AbstractAggregator(ABC):
    @abstractmethod
    def aggregate(self, input_data: AggregatorInput) -> AggregatorOutput:

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
