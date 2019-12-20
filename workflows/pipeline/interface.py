from dataclasses import dataclass
from typing import List, Union, Dict
import logging
import numpy as np
from aggregators import AbstractAggregator
from api.sequence_creator import create_sequence_3d
from processors import StandardDataFormat, AbstractProcessor
from workflows.utils import get_qualified_name

logger = logging.getLogger()


# python 3.8+
# class InputOutputField(TypedDict):
#     inputField: str
#     outputField: str
InputOutputField = Dict


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
        self.grouped_data = create_sequence_3d(features=data.data, n_sequence=sequence)
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
        logger.debug("execute multi aggregation with instances of={0}".format(
            list(map(get_qualified_name, self.instances)))
        )
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
            result = aggregator.instance.aggregate(grouped_data=grouped_data)
            collector.hstack_bottom(fields_out, result.metrics)

        return processor_input.modify_copy(
            labels=processor_input.labels + collector.labels,
            data=collector.result
        )


ProcessorOrMultiAggregation = Union[MultiAggregation, AbstractProcessor]


@dataclass
class PipelineWorkflow:
    pipelines: List[ProcessorOrMultiAggregation]

    def execute(self, input_data: StandardDataFormat) -> StandardDataFormat:
        for pipe in self.pipelines:
            input_data = pipe.process(input_data)
        return input_data