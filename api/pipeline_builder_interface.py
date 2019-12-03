from dataclasses import dataclass
from typing import List, TypedDict, Union, Optional
from aggregators import AbstractAggregator
from datasources import AbstractDatasourceAdapter
from groupers import AbstractGrouper
from processors import AbstractProcessor


class InputOutputField(TypedDict):
    inputField: str
    outputField: str


@dataclass
class SingleAggregationConfig:
    sequence: int
    instance: AbstractAggregator
    generate: List[InputOutputField]


@dataclass
class MultiAggregationConfig:
    sequence: int
    instances: List[SingleAggregationConfig]


Pipeline = List[Union[AbstractProcessor, MultiAggregationConfig]]


@dataclass
class AnalyzerConfig:
    group_by: List[AbstractGrouper]
    aggregators: List[AbstractAggregator]


@dataclass
class BuildConfig:
    source: AbstractDatasourceAdapter
    fields: List[str]
    pipeline: Optional[Pipeline]
    analyzer: Optional[AnalyzerConfig]
