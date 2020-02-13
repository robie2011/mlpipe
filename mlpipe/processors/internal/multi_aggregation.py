import logging
from dataclasses import dataclass
from typing import List, Union
import numpy as np
from mlpipe.aggregators.abstract_aggregator import AbstractAggregator
from mlpipe.dsl_interpreter.descriptions import InputOutputField
from mlpipe.processors.interfaces import AbstractProcessor
from mlpipe.processors.sequence3d import Sequence3d
from mlpipe.processors.standard_data_format import StandardDataFormat
from mlpipe.workflows.utils import get_qualified_name

module_logger = logging.getLogger(__name__)

# python 3.8+
# class InputOutputField(TypedDict):
#     inputField: str
#     outputField: str


@dataclass
class MissingFields(BaseException):
    fields: List[str]


@dataclass
class MissingFieldsForLogic(MissingFields):
    logic: object

    def __str__(self):
        return "For logic: \n\t{0} \n\tfollowing fields are missing as input variable: {1}".format(
            self.logic,
            ", ".join(self.fields)
        )


class MultiAggregationDataFormat:
    def __init__(self, data: StandardDataFormat, sequence: int):
        self.grouped_data = Sequence3d.create_sequence_3d(features=data.data, n_sequence=sequence)
        self.ix_by_label = {data.labels[i]: i for i in range(len(data.labels))}

    def get_partial_data(self, fields: List[str]) -> np.ndarray:
        non_existing_fields = list(filter(lambda x: x not in self.ix_by_label, fields))
        if len(non_existing_fields) > 0:
            raise MissingFields(fields=non_existing_fields)

        column_ids = [self.ix_by_label[f] for f in fields]
        return self.grouped_data[:, :, column_ids]


class MultiAggregationResultCollector:
    def __init__(self, data2d: np.ndarray):
        self.result: np.ndarray = data2d
        self.labels = []

    def hstack_bottom(self, labels: List[str], output: np.ndarray):
        rows_missing = self.result.shape[0] - output.shape[0]
        dummy_data = np.full((rows_missing, output.shape[1]), fill_value=np.nan, dtype='float')
        output = np.vstack((dummy_data, output))

        self.result = np.hstack((self.result, output))
        self.labels += labels


@dataclass
class SingleAggregationConfig:
    sequence: int
    instance: AbstractAggregator
    generate: List[InputOutputField]


@dataclass
class MultiAggregation(AbstractProcessor):
    sequence: int
    instances: List[SingleAggregationConfig]

    def process(self, processor_input: StandardDataFormat) -> StandardDataFormat:
        module_logger.debug("execute multi aggregation with instances of={0}".format(
            list(map(lambda agg: get_qualified_name(agg.instance), self.instances))
        ))

        map(lambda agg: type(agg.instance).__name__, self.instances)

        aggregation_data = MultiAggregationDataFormat(data=processor_input, sequence=self.sequence)
        collector = MultiAggregationResultCollector(processor_input.data)
        for aggregator in self.instances:
            fields_in = [inputOutputField['inputField'] for inputOutputField in aggregator.generate]
            fields_out = [inputOutputField['outputField'] for inputOutputField in aggregator.generate]

            try:
                grouped_data = aggregation_data.get_partial_data(fields=fields_in)
            except MissingFields as e:
                raise MissingFieldsForLogic(fields=e.fields, logic=aggregator)

            grouped_data.flags.writeable = False
            module_logger.debug("  > run aggreagtion with {0}".format(get_qualified_name(aggregator.instance)))
            result = aggregator.instance.aggregate(grouped_data=grouped_data)
            collector.hstack_bottom(fields_out, result.metrics)

        return processor_input.modify_copy(
            labels=processor_input.labels + collector.labels,
            data=collector.result
        )


ProcessorOrMultiAggregation = Union[MultiAggregation, AbstractProcessor]
