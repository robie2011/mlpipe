from dataclasses import dataclass
from typing import TypedDict, List
from api.pipline_builder import Pipeline, MultiAggregationConfig
import numpy as np
from api.sequence_creator import create_sequence_3d
from datasources import AbstractDatasourceAdapter
from processors import StandardDataFormat, AbstractProcessor
import logging


logger = logging.getLogger("pipeline.executor")


def execute_pipeline(source: AbstractDatasourceAdapter, fields: List[str], pipeline: Pipeline):
    logger.debug("connect to datasource with adapter {0}".format(source.__class__))
    canConnect = source.test()
    if canConnect is not True:
        raise Exception("Can not connect to source: ", canConnect)

    data = source.fetch()

    logger.debug("field descriptions: {0}".format(", ".join(fields)))
    logger.debug("fields from source: {0}".format(", ".join(data.labels)))
    logger.debug("rename field names according to alias")
    for xs in [x.split(" as ") for x in fields]:
        # note: if no alias was set with "as"-keyword
        # original and alias name will be equal
        name_original = xs[0].strip()
        name_alias = xs[-1].strip()

        ix = data.labels.index(name_original)
        data.labels[ix] = name_alias

    for pipe in pipeline:
        if isinstance(pipe, AbstractProcessor):
            processor: AbstractProcessor = pipe
            data = processor.process(data)
        elif isinstance(pipe, MultiAggregationConfig):
            config: MultiAggregationConfig = pipe
            data = run_multi_aggregation(data2d=data, config=config)
        else:
            raise Exception("Unknown Pipe Type", pipe)
    return data


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

# note:
#   we have our data in a standardized format
#   than we need special formatted data for different pipe: processor, windowed feature extractor, feature extractor
#   output of pipe should be merged to basic format
#   todo: output of windowed feature is smaller than the original. Missing data should be filled somehow
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


def run_multi_aggregation(data2d: StandardDataFormat, config: MultiAggregationConfig) -> StandardDataFormat:
    logger.debug("execute multi aggregation w/ {0}".format(
        list(map(lambda x: x.instance, config.instances)))
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
