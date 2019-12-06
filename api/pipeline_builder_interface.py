from dataclasses import dataclass
from typing import List, Union, Optional
from workflows.analyzers.analyzer_workflow import AnalyzerWorkflow
from datasources import AbstractDatasourceAdapter
from processors import AbstractProcessor
import logging

from workflows.pipeline.interface import MultiAggregation

logger = logging.getLogger()

Pipeline = List[Union[AbstractProcessor, MultiAggregation]]


@dataclass
class BuildConfig:
    source: AbstractDatasourceAdapter
    fields: List[str]
    pipeline: Optional[Pipeline]
    analyzer: Optional[AnalyzerWorkflow]
