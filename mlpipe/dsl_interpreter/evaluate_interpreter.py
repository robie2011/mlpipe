import logging
from typing import Dict

from mlpipe.config.training_project import TrainingProject
from mlpipe.datasources.internal.cached_datasource import CachedDatasource
from mlpipe.dsl_interpreter.descriptions import ExecutionModes
from mlpipe.workflows.evaluate.evaluate_workflow_manager import EvaluateWorkflowManager
from mlpipe.pipeline.pipeline_builder import build_pipeline_executor

module_logger = logging.getLogger(__name__)


def _create_workflow_evaluate(description: Dict, execution_mode: ExecutionModes) -> EvaluateWorkflowManager:
    name, session_id = description['name'], description['session']
    project = TrainingProject(name=name, session_id=session_id, create=False)
    model = project.model

    desc_merged = project.description.copy()
    desc_merged['source'] = description['source']

    source_adapter = CachedDatasource(desc_merged['source'])

    description_pipeline = []
    description_pipeline += desc_merged.get('pipelinePrimary', [])
    description_pipeline += desc_merged.get('pipelineSecondary', [])
    module_logger.info(f"filters found: {len(description_pipeline)}")

    pipeline_executor = build_pipeline_executor(descriptions=description_pipeline, execution_mode=execution_mode)

    return EvaluateWorkflowManager(
        description=desc_merged,
        data_adapter=source_adapter,
        pipeline_executor=pipeline_executor,
        model=model,
        pipeline_states=project.states
    )
