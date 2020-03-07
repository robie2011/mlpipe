from dataclasses import dataclass
from typing import List, Dict

# python 3.8+
# class AnalyzeDescription(TypedDict):
#     groupBy: List[str]
#     metrics: List[ClassDescription]
AnalyzeDescription = Dict


@dataclass
class AnalyticsResultMeta:
    sensors: List[str]
    metrics: List[str]
    groupers: List[str]
    groupToPartitionerToPartition: List[List[int]]
    prettyGroupnames: List[List[str]]
    metricsAggregationFunc: List[str]


@dataclass
class AnalyticsResult:
    meta: AnalyticsResultMeta
    groupToMetricToSensorToMeasurement: List[List[List[float]]]
    metrics_datastructure_help = "shape is (groups, aggregations, sensors)"
