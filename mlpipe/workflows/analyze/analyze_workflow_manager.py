from dataclasses import dataclass
from typing import List

import numpy as np

from mlpipe.aggregators.abstract_aggregator import AbstractAggregator
from mlpipe.groupers import AbstractGrouper
from mlpipe.workflows.utils import get_qualified_name, get_class_name
from .helper import create_grouped_data, group_by_multi_columns
from .interface import AnalyticsResultMeta, AnalyticsResult
from ..abstract_workflow_manager import AbstractWorkflowManager


@dataclass
class AnalyzeWorkflowManager(AbstractWorkflowManager):
    groupers: List[AbstractGrouper]
    metrics: List[AbstractAggregator]

    def run(self):
        logger = self.logger
        logger.info(f"groupers: {','.join(map(get_qualified_name, self.groupers))}")
        logger.info(f"metrics: {','.join(map(get_qualified_name, self.metrics))}")

        source_data = self.data_adapter.get()
        input_data = self.pipeline_executor.execute(source_data)

        data_partitions = np.array([
            g.group(timestamps=input_data.timestamps)
            for g in self.groupers
        ]).T

        features = input_data.data

        # combigroup
        cgroups = group_by_multi_columns(data_partitions)

        # note: groups can have different size
        n_max_group_members = np.max(np.fromiter(map(lambda x: x.indexes.shape[0], cgroups), dtype='int'))
        grouped_data = create_grouped_data(cgroups, n_max_group_members, features)

        output = np.full(
            (len(cgroups), len(self.metrics), features.shape[1]),
            fill_value=np.nan,
            dtype='float64'
        )

        for i, aggreagtor in enumerate(self.metrics):
            logger.debug("aggregate using: {0}".format(get_qualified_name(aggreagtor)))
            output[:, i, :] = aggreagtor.aggregate(grouped_data=grouped_data)

        group_ids = np.array(list(map(lambda x: x.group_id, cgroups))).tolist()

        meta = AnalyticsResultMeta(
            sensors=input_data.labels,
            metrics=list(map(get_class_name, self.metrics)),
            groupers=list(map(get_class_name, self.groupers)),
            groupToPartitionerToPartition=group_ids,
            prettyGroupnames=list(map(lambda x: x.get_pretty_group_names(), self.groupers)),
            metricsAggregationFunc=list(map(lambda x: x.javascript_group_aggregation(), self.metrics))
        )

        return AnalyticsResult(
            meta=meta,
            groupToMetricToSensorToMeasurement=output.tolist()
        )
