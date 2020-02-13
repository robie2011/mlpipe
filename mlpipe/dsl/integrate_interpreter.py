import logging
from typing import Dict

from mlpipe.config.training_project import TrainingProject
from mlpipe.datasources.internal.cached_datasource import CachedDatasource
from mlpipe.dsl.instance_creator import create_output_adapter
from mlpipe.workflows.evaluate.prediction_evaluators import prediction_evaluators
from mlpipe.workflows.integrate.integration_workflow_manager import IntegrationWorkflowManager
from mlpipe.workflows.pipeline.pipeline_builder import build_pipeline_executor

module_logger = logging.getLogger(__name__)


def _create_workflow_integrate(description: Dict) -> IntegrationWorkflowManager:
    name, session_id = description['name'], description['session']
    project = TrainingProject(name=name, session_id=session_id, create=False)
    model = project.model

    desc_merged = project.description.copy()
    desc_merged['source'] = description['source']
    prediction_type = desc_merged['model']['predictionType']
    execution_limit = description.get("limitExecution", -1)

    try:
        evaluator = prediction_evaluators.get(prediction_type)
    except KeyError as e:
        raise ValueError(f"Prediction type '{prediction_type}' not implemented!")

    source_adapter = CachedDatasource(desc_merged['source'])
    output_adapter = create_output_adapter(description['output'])

    description_pipeline = []
    description_pipeline += desc_merged.get('pipelinePrimary', [])
    description_pipeline += desc_merged.get('pipelineSecondary', [])
    module_logger.info(f"pipes found: {len(description_pipeline)}")

    pipeline_executor = build_pipeline_executor(descriptions=description_pipeline)

    return IntegrationWorkflowManager(
        description=desc_merged,
        data_adapter=source_adapter,
        pipeline_executor=pipeline_executor,
        model=model,
        evaluator=evaluator,
        pipeline_states=project.states,
        name=name,
        session_id=session_id,
        frequency_minutes=description['executionFrequencyMinutes'],
        output_adapter=output_adapter,
        limit_execution=execution_limit
    )
