from dataclasses import dataclass
from typing import List
import numpy as np
from aggregators import AbstractAggregator
from .helper import create_np_group_data
from workflows.analyzers.interface import AnalyticsResultMeta, AnalyticsResult
from groupers import AbstractGrouper, group_by_multi_columns
from processors import StandardDataFormat
from workflows.utils import get_qualified_name
import logging

logger = logging.getLogger()


@dataclass
class AnalyzerWorkflow:
    group_by: List[AbstractGrouper]
    aggregators: List[AbstractAggregator]

    def run(self, input_data: StandardDataFormat):
        logger.info("start analysis")
        logger.debug("grouping: {0}".format(", ".join(map(get_qualified_name, self.group_by))))
        data_partitions = np.array([
            g.group(timestamps=input_data.timestamps, raw_data=input_data.data)
            for g in self.group_by
        ]).T

        groups = group_by_multi_columns(data_partitions)
        n_groups = len(groups)
        n_max_group_members = np.max(np.fromiter(map(lambda x: x.indexes.shape[0], groups), dtype='int'))
        grouped_data = create_np_group_data(groups, n_groups, n_max_group_members, input_data.data)

        output = np.full(
            (n_groups, len(self.aggregators), input_data.data.shape[1]),
            fill_value=np.nan,
            dtype='float64'
        )

        for i in range(len(self.aggregators)):
            aggreagtor = self.aggregators[i]
            logger.debug("aggregate using: {0}".format(get_qualified_name(aggreagtor)))
            output[:, i, :] = aggreagtor.aggregate(grouped_data=grouped_data).metrics

        group_ids = np.array(list(map(lambda x: list(x.group_id), groups))).tolist()
        meta = AnalyticsResultMeta(
            sensors=input_data.labels,
            metrics=list(map(get_qualified_name, self.aggregators)),
            groupers=list(map(get_qualified_name, self.group_by)),
            groups=group_ids,
            prettyGroupnames=list(map(lambda x: x.get_pretty_group_names() , self.group_by))
        )

        return AnalyticsResult(
            meta=meta,
            metrics=output.tolist()
        )