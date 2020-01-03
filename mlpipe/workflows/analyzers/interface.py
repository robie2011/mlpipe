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
    groups: List[List[int]]
    prettyGroupnames: List[str]


@dataclass
class AnalyticsResult:
    meta: AnalyticsResultMeta
    metrics: List[List[float]]
