import logging
from typing import Dict

from mlpipe.datasources.internal.cached_datasource import CachedDatasource
from mlpipe.workflows.pipeline.pipeline_builder import build_pipeline_executor
from mlpipe.workflows.train.train_workflow_manager import TrainWorkflowManager

module_logger = logging.getLogger(__name__)


def _create_workflow_training(description: Dict):
    source_adapter = CachedDatasource(description['source'])

    description_pipeline = []
    description_pipeline += description.get('pipelinePrimary', [])
    description_pipeline += description.get('pipelineSecondary', [])
    module_logger.info(f"pipes found: {len(description_pipeline)}")
    pipeline_executor = build_pipeline_executor(descriptions=description_pipeline)

    return TrainWorkflowManager(
        description=description,
        name=description['name'],
        data_adapter=source_adapter,
        pipeline_executor=pipeline_executor)