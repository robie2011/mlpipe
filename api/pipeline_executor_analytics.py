from dataclasses import dataclass
from typing import List
import numpy as np
from api.pipeline_builder_interface import AnalyzerConfig
from groupers import group_by_multi_columns, CombinedGroup, AbstractGrouper
from processors import StandardDataFormat
import logging
from utils import get_qualified_name
import json

logger = logging.getLogger("pipeline.executor.analytics")


@dataclass
class AnalyticsResultMeta:
    sensors: List[str]
    metrics: List[str]
    groupers: List[str]
    groups: List[List[int]]
    prettyGroupnames: List[str]

@dataclass
class AnalyticsResult:
    meta: AnalyticsResultMeta
    metrics: List[List[float]]


def execute_analytics(input_data: StandardDataFormat, config: AnalyzerConfig):
    logger.info("start analysis")
    logger.debug("grouping: {0}".format(", ".join(map(get_qualified_name, config.group_by))))
    data_partitions = np.array([
        g.group(timestamps=input_data.timestamps, raw_data=input_data.data)
        for g in config.group_by
    ]).T

    groups = group_by_multi_columns(data_partitions)
    n_groups = len(groups)
    n_max_group_members = np.max(np.fromiter(map(lambda x: x.indexes.shape[0], groups), dtype='int'))
    grouped_data = create_np_group_data(groups, n_groups, n_max_group_members, input_data.data)

    output = np.full(
        (n_groups, len(config.aggregators), input_data.data.shape[1]),
        fill_value=np.nan,
        dtype='float64'
    )

    for i in range(len(config.aggregators)):
        aggreagtor = config.aggregators[i]
        logger.debug("aggregate using: {0}".format(get_qualified_name(aggreagtor)))
        output[:, i, :] = aggreagtor.aggregate(grouped_data=grouped_data).metrics

    group_ids = np.array(list(map(lambda x: list(x.group_id), groups))).tolist()
    meta = AnalyticsResultMeta(
        sensors=input_data.labels,
        metrics=list(map(get_qualified_name, config.aggregators)),
        groupers=list(map(get_qualified_name, config.group_by)),
        groups=group_ids,
        prettyGroupnames=list(map(lambda x: x.get_pretty_group_names() , config.group_by))
    )

    return AnalyticsResult(
        meta=meta,
        metrics=output.tolist()
    )


def create_np_group_data(groups, n_groups, n_max_group_members, raw_data_only):
    # note: numpy currently do not support NaN for integer type array
    # instead of nan we will get a very big negative value
    # therefore we need to drop negative integers later
    # see also: https://stackoverflow.com/questions/12708807/numpy-integer-nan

    # note: true values for masked array means block that value
    logger.debug("create grouped indexes")
    grouped_indexes = np.ma.zeros((n_groups, n_max_group_members), dtype='int')
    grouped_indexes.mask = np.ones((n_groups, n_max_group_members), dtype='int')
    for i in range(len(groups)):
        g: CombinedGroup = groups[i]
        n_current_group_size = g.indexes.shape[0]
        grouped_indexes[i, :n_current_group_size] = g.indexes
        grouped_indexes.mask[i, :n_current_group_size] = False

    n_sensors = raw_data_only.shape[1]
    grouped_data = np.ma.zeros(
        (n_groups, n_max_group_members, n_sensors),
        fill_value=np.nan,
        dtype='float64')
    grouped_data.mask = grouped_indexes.mask

    logger.debug("grouping indexes/data")
    for group_id in range(n_groups):
        _mask = np.invert(grouped_indexes.mask[group_id, :])
        indexes = grouped_indexes[group_id][_mask]
        n_samples = len(indexes)
        grouped_data[group_id, :n_samples] = raw_data_only[indexes, :]
    return grouped_data
