from dataclasses import dataclass
from typing import List, Union, Dict
import logging
import numpy as np
from mlpipe.aggregators import AbstractAggregator
from mlpipe.api.sequence_creator import create_sequence_3d
from mlpipe.processors import StandardDataFormat, AbstractProcessor
from mlpipe.workflows.utils import get_qualified_name


module_logger = logging.getLogger(__name__)


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


@dataclass
class PipelineWorkflow:
    pipelines: List[ProcessorOrMultiAggregation]

    def execute(self, input_data: StandardDataFormat) -> StandardDataFormat:
        for pipe in self.pipelines:
            module_logger.debug("execute pipe: {0}".format(type(pipe).__name__))
            input_data_after = pipe.process(input_data)
            self._analyze(input_data_before=input_data, input_data_after=input_data_after, processor=pipe)
            input_data = input_data_after
        return input_data

    def _analyze(
            self,
            input_data_before: StandardDataFormat,
            input_data_after: StandardDataFormat,
            processor: ProcessorOrMultiAggregation):
        n_total = input_data_before.data.shape[0]
        n_total_new = input_data_after.data.shape[0]
        n_remove = n_total - n_total_new
        p_removed = n_remove / n_total
        p_usual_cleaning = .1

        if n_remove != 0:
            module_logger.debug("Row size updated: change: {0:,} from total {1:,}. Remaining {2:,}. Processed by {3}".format(
                (-1) * n_remove,
                n_total,
                n_total_new,
                get_qualified_name(processor)
            ))

            if p_removed > p_usual_cleaning:
                module_logger.warning("unusual behaviour: {0}% of source rows were removed by {1}".format(
                    p_removed * 100,
                    get_qualified_name(processor)
                ))
