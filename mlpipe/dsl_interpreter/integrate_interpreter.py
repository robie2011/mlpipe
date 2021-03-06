import logging
from typing import Dict

from mlpipe.config.training_project import TrainingProject
from mlpipe.dsl_interpreter.descriptions import ExecutionModes
from mlpipe.dsl_interpreter.instance_creator import create_output_adapter, create_source_adapter
from mlpipe.workflows.integrate.integrate_workflow_manager import IntegrateWorkflowManager
from mlpipe.pipeline.pipeline_builder import build_pipeline_executor

module_logger = logging.getLogger(__name__)


def _create_workflow_integrate(description: Dict, execution_mode: ExecutionModes) -> IntegrateWorkflowManager:
    name, session_id = description['name'], description['session']
    project = TrainingProject(name=name, session_id=session_id, create=False)
    model = project.model

    desc_merged = project.description.copy()
    desc_merged['source'] = description['source']
    execution_limit = description.get("limitExecution", -1)

    source_adapter = create_source_adapter(desc_merged['source'])
    output_adapter = create_output_adapter(description['integrate']['output'])

    description_pipeline = []
    description_pipeline += desc_merged.get('pipelinePrimary', [])
    description_pipeline += desc_merged.get('pipelineSecondary', [])
    module_logger.info(f"filters found: {len(description_pipeline)}")

    pipeline_executor = build_pipeline_executor(descriptions=description_pipeline, execution_mode=execution_mode)

    return IntegrateWorkflowManager(
        description=desc_merged,
        data_adapter=source_adapter,
        pipeline_executor=pipeline_executor,
        model=model,
        pipeline_states=project.states,
        name=name,
        session_id=session_id,
        frequency_minutes=description['integrate'].get('frequencyMin', 1),
        output_adapter=output_adapter,
        limit_execution=execution_limit
    )
