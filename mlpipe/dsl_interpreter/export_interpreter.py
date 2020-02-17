from typing import Dict
from mlpipe.datasources.internal.cached_datasource import CachedDatasource
from mlpipe.workflows.analyze.data_export_workflow_manager import DataExportWorkflowManager
from mlpipe.workflows.pipeline.pipeline_builder import build_pipeline_executor


def _create_workflow_analyze_export(description: Dict):
    source_adapter = CachedDatasource(description['source'])
    description_pipeline = []

    if description['@pipelinePrimary']:
        description_pipeline += description.get('pipelinePrimary', [])

    if description['@pipelineSecondary']:
        description_pipeline += description.get('pipelineSecondary', [])

    pipeline_executor = build_pipeline_executor(descriptions=description_pipeline)

    return DataExportWorkflowManager(
        description=description,
        data_adapter=source_adapter,
        pipeline_executor=pipeline_executor)