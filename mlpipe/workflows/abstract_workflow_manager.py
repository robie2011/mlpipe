from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict
from mlpipe.datasources.internal.cached_datasource import CachedDatasource
from mlpipe.mixins.logger_mixin import InstanceLoggerMixin
from mlpipe.workflows.pipeline.pipeline_executor import PipelineExecutor
from mlpipe.workflows.pipeline.standard_dataflow_analyzer import StandardDataflowAnalyzer
from mlpipe.workflows.utils import get_qualified_name


@dataclass
class AbstractWorkflowManager(ABC, InstanceLoggerMixin):
    description: Dict
    data_adapter: CachedDatasource
    pipeline_executor: PipelineExecutor

    @abstractmethod
    def run(self):
        pass

    def _reset_pipeline_stats(self):
        pipeline_stats = StandardDataflowAnalyzer()
        self.pipeline_executor.set_on_pipe_start_handler(pipeline_stats.before_pipe_handler)
        self.pipeline_executor.set_on_pipe_end_handler(pipeline_stats.after_pipe_handler)

    def __post_init__(self):
        self.get_logger().info(f"using workflow manager {get_qualified_name(self)}")
        self._reset_pipeline_stats()
