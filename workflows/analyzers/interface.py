from dataclasses import dataclass
from typing import TypedDict, List
from workflows.interface import ClassDescription


class AnalyzeDescription(TypedDict):
    groupBy: List[str]
    metrics: List[ClassDescription]


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