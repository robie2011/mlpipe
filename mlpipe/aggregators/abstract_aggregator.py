from abc import abstractmethod
from typing import List

import numpy as np

from mlpipe.dsl_interpreter.descriptions import InputOutputField
from mlpipe.processors.interfaces import AbstractProcessor
from .aggregator_output import AggregatorOutput
from ..processors.internal.multi_aggregation_data_format import MultiAggregationDataFormat
from ..processors.internal.multi_aggregation_fields import MissingFields, MissingFieldsForLogic
from ..processors.internal.multi_aggregation_result_collector import MultiAggregationResultCollector
from ..processors.standard_data_format import StandardDataFormat


class AbstractAggregator(AbstractProcessor):
    def __init__(self, sequence: int, generate: List[InputOutputField]):
        self.sequence = sequence
        self.generate = generate

    def _process2d(self, processor_input: StandardDataFormat) -> StandardDataFormat:
        collector = MultiAggregationResultCollector(data=processor_input.data, labels=processor_input.labels)
        aggregation_data = MultiAggregationDataFormat(data=processor_input, sequence=self.sequence)

        fields_out, grouped_data = self.select_data(aggregation_data, fallback_labels=processor_input.labels)
        result = self.aggregate(grouped_data=grouped_data)

        collector.hstack_bottom(fields_out, result)

        return processor_input.modify_copy(
            labels=collector.labels,
            data=collector.data
        )

    def select_data(self, aggregation_data: MultiAggregationDataFormat, fallback_labels: List[str]):
        if not self.generate:
            # case 1: we do not have any information about input/output
            fields_in = fallback_labels
            fields_out = [f"{n}${self.__name__}" for n in fallback_labels]
        else:
            # case 2: fields_in are defined and fields_out are optionally defined
            #   if field_out not defined than generate a name
            fields_in, fields_out = [], []
            for f in self.generate:
                fields_in.append(f['inputField'])
                fields_out.append(f.get('outputField', f"{f['inputField']}${self.__name__}"))

        try:
            grouped_data = aggregation_data.get_partial_data(fields=fields_in)
        except MissingFields as e:
            raise MissingFieldsForLogic(fields=e.fields, logic=self)
        grouped_data.flags.writeable = False
        return fields_out, grouped_data

    @abstractmethod
    def aggregate(self, grouped_data: np.ndarray) -> AggregatorOutput:

        """
        input: 3D-Numpy Array
        first axis represents sequence id or group id
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
