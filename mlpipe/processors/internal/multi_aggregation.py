import logging
from dataclasses import dataclass
from typing import List, Union
from mlpipe.aggregators.abstract_aggregator import AbstractAggregator
from mlpipe.processors.internal.multi_aggregation_fields import MissingFields, MissingFieldsForLogic
from mlpipe.dsl_interpreter.descriptions import InputOutputField
from mlpipe.processors.interfaces import AbstractProcessor
from mlpipe.processors.internal.multi_aggregation_data_format import MultiAggregationDataFormat
from mlpipe.processors.internal.multi_aggregation_result_collector import MultiAggregationResultCollector
from mlpipe.processors.standard_data_format import StandardDataFormat
from mlpipe.workflows.utils import get_qualified_name

module_logger = logging.getLogger(__name__)


@dataclass
class MultiAggregation(AbstractProcessor):
    sequence: int
    instances: List[AbstractAggregator]

    def process(self, processor_input: StandardDataFormat) -> StandardDataFormat:
        module_logger.debug("execute multi aggregation with instances of={0}".format(
            list(map(lambda agg: get_qualified_name(agg), self.instances))
        ))

        aggregation_data = MultiAggregationDataFormat(data=processor_input, sequence=self.sequence)
        collector = MultiAggregationResultCollector(data=processor_input.data, labels=processor_input.labels)
        for aggregator in self.instances:
            MultiAggregation.run_aggregator(aggregation_data, aggregator, collector)

        return processor_input.modify_copy(
            labels=collector.labels,
            data=collector.data
        )

    @staticmethod
    def run_aggregator(aggregation_data, aggregator, collector):
        fields_in = [inputOutputField['inputField'] for inputOutputField in aggregator.generate]
        fields_out = [inputOutputField['outputField'] for inputOutputField in aggregator.generate]
        try:
            grouped_data = aggregation_data.get_partial_data(fields=fields_in)
        except MissingFields as e:
            raise MissingFieldsForLogic(fields=e.fields, logic=aggregator)
        grouped_data.flags.writeable = False
        module_logger.debug("  > run aggreagtion with {0}".format(get_qualified_name(aggregator)))
        result = aggregator.aggregate(grouped_data=grouped_data)
        collector.hstack_bottom(fields_out, result)

    @staticmethod
    def from_aggregator(aggregator: AbstractAggregator):
        return MultiAggregation(sequence=aggregator.sequence, instances=[aggregator])


ProcessorOrMultiAggregation = Union[MultiAggregation, AbstractProcessor]
