from dataclasses import dataclass
from typing import Optional

from api.pipeline_executor_analytics import execute_analytics, AnalyticsResult
from api.pipeline_executor_interface import \
    MultiAggregationDataFormat, \
    MultiAggregationResultCollector, \
    MissingFieldsForLogic, MissingFields
from api.pipline_builder import MultiAggregationConfig, BuildConfig
from groupers import CombinedGroup
from processors import StandardDataFormat, AbstractProcessor
import logging
import numpy as np
from utils import get_qualified_name

logger = logging.getLogger("pipeline.executor")


@dataclass
class ExecutionResult:
    pipeline_data: StandardDataFormat
    analytics: Optional[AnalyticsResult]


def execute_pipeline(build_config: BuildConfig):
    logger.info("connect to datasource with adapter {0}".format(build_config.source.__class__))
    canConnect = build_config.source.test()
    if canConnect is not True:
        raise Exception("Can not connect to source: ", canConnect)

    data = build_config.source.fetch()

    logger.debug("field descriptions: {0}".format(", ".join(build_config.fields)))
    logger.debug("fields from source: {0}".format(", ".join(data.labels)))
    logger.debug("rename field names according to alias")
    for xs in [x.split(" as ") for x in build_config.fields]:
        # note: if no alias was set with "as"-keyword
        # original and alias name will be equal
        name_original = xs[0].strip()
        name_alias = xs[-1].strip()

        ix = data.labels.index(name_original)
        data.labels[ix] = name_alias

    for pipe in build_config.pipeline:
        if isinstance(pipe, AbstractProcessor):
            processor: AbstractProcessor = pipe
            data = processor.process(data)
        elif isinstance(pipe, MultiAggregationConfig):
            config: MultiAggregationConfig = pipe
            data = execute_multi_aggregation(data2d=data, config=config)
        else:
            raise Exception("Unknown Pipe Type", pipe)

    analytics_data = None
    if build_config.analyzer:
        analytics_data = execute_analytics(input_data=data, config=build_config.analyzer)
        # todo: see test/test_analytics_chain.py, line: 94ff

    return ExecutionResult(
        pipeline_data=data,
        analytics=analytics_data
    )


def execute_multi_aggregation(data2d: StandardDataFormat, config: MultiAggregationConfig) -> StandardDataFormat:
    logger.debug("execute multi aggregation with instances of={0}".format(
        list(map(get_qualified_name, config.instances)))
    )
    aggregation_data = MultiAggregationDataFormat(data=data2d, sequence=config.sequence)
    collector = MultiAggregationResultCollector(data2d.data)
    for aggregator in config.instances:
        fields_in = [f['inputField'] for f in aggregator.generate]
        fields_out = [f['outputField'] for f in aggregator.generate]

        try:
            grouped_data = aggregation_data.get_partial_data(fields=fields_in)
        except MissingFields as e:
            raise MissingFieldsForLogic(fields=e.fields, logic=aggregator)

        grouped_data.flags.writeable = False
        result = aggregator.instance.aggregate(grouped_data=grouped_data)
        collector.hstack_bottom(fields_out, result.metrics)

    return StandardDataFormat(
        labels=data2d.labels + collector.labels,
        data=collector.result,
        timestamps=data2d.timestamps
    )

