from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict

from mlpipe.datasources.abstract_datasource_adapter import AbstractDatasourceAdapter
from mlpipe.datasources.internal.cached_datasource import CachedDatasource
from mlpipe.utils.logger_mixin import InstanceLoggerMixin
from mlpipe.pipeline.pipeline_executor import PipelineExecutor
from mlpipe.pipeline.standard_dataflow_analyzer import StandardDataflowAnalyzer
from mlpipe.workflows.utils import get_qualified_name


@dataclass
class AbstractWorkflowManager(ABC, InstanceLoggerMixin):
    description: Dict
    data_adapter: [AbstractDatasourceAdapter, CachedDatasource]
    pipeline_executor: PipelineExecutor

    @abstractmethod
    def run(self):
        pass

    def _reset_pipeline_stats(self):
        self.pipeline_executor.flow_analyzer = StandardDataflowAnalyzer()
        self.pipeline_executor.flow_analyzer.init_flow(self.description)

    def __post_init__(self):
        self.logger.info(f"using workflow manager {get_qualified_name(self)}")
        if self.pipeline_executor:
            self._reset_pipeline_stats()
