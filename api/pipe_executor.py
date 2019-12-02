from typing import TypedDict, List
from api.pipline_builder import Pipeline, MultiAggregationConfig
import numpy as np
from api.sequence_creator import create_sequence_3d
from processors import StandardDataFormat, AbstractProcessor



def get_label_index(labels_given: List[str], labels_selection: List[str]) -> List[int]:
    ix_by_label = {labels_given[i]: i for i in range(len(labels_given))}
    return [ix_by_label[x] for x in labels_selection]


def run(source, pipline: Pipeline):
    pass

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
        column_ids = [self.ix_by_label[f] for f in fields]
        return self.grouped_data[:, :, column_ids]


class MultiAggregationResultCollector:
    def __init__(self, data2d: np.ndarray):
        self.result: np.ndarray = data2d
        self.labels = []

    def hstack_bottom(self, labels: List[str], output: np.ndarray):
        rows_missing = self.result.shape[0] - output.shape[0]
        dummy_data = np.full((rows_missing, 1), fill_value=np.nan, dtype='float')
        output = np.vstack((dummy_data, output))

        self.result = np.hstack((self.result, output))
        self.labels += labels


# todo: write test
def run_multi_aggregation(data2d: StandardDataFormat, definition: MultiAggregationConfig) -> StandardDataFormat:
    aggregation_data = MultiAggregationDataFormat(data=data2d, sequence=definition.sequence)
    collector = MultiAggregationResultCollector(data2d.data)
    for aggregator in definition.instances:
        fields_in = [f['inputField'] for f in aggregator.generate]
        fields_out = [f['outputField'] for f in aggregator.generate]

        grouped_data = aggregation_data.get_partial_data(fields=fields_in)
        grouped_data.flags.writeable = False
        result = aggregator.instance.aggregate(grouped_data=grouped_data)
        collector.hstack_bottom(fields_out, result.metrics)

    return StandardDataFormat(
        labels=data2d.labels + collector.labels,
        data=collector.result,
        timestamps=data2d.timestamps
    )


# note: run_processor no required
# def run_feature_extractor(data2d: StandardDataFormat, definition: FeatureExtractorConfig) -> StandardDataFormat:
#     ix_columns_in = get_label_index(data2d.labels, [g['inputField'] for g in definition.generate])
#     fields_out = [g['outputField'] for g in definition.generate]
#     data_output = definition.instance.extract(
#         timestamps=data2d.timestamps,
#         features=data2d.data[:, ix_columns_in])
#
#     return StandardDataFormat(
#         labels=data2d.labels + fields_out,
#         data=np.hstack((data2d.data, data_output)),
#         timestamps=data2d.timestamps
#     )
