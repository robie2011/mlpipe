import logging
from typing import Dict

from mlpipe.aggregators import AbstractAggregator
from mlpipe.datasources.internal.cached_datasource import CachedDatasource
from mlpipe.dsl import _get_descriptions_name
from mlpipe.groupers import AbstractGrouper
from mlpipe.workflows.analyze.analyze_workflow_manager import AnalyzeWorkflowManager
from mlpipe.workflows.pipeline.pipeline_builder import build_pipeline_executor
from mlpipe.workflows.utils import create_instance, get_component_config

module_logger = logging.getLogger(__name__)


def _create_workflow_analyze(description: Dict):
    source_adapter = CachedDatasource(description['source'])

    description_pipeline = []
    description_pipeline += description.get('pipelinePrimary', [])

    str_pipes = ", ".join(_get_descriptions_name(description_pipeline))
    module_logger.info(f"pipes found {len(description_pipeline)}: {str_pipes}")
    pipeline_executor = build_pipeline_executor(descriptions=description_pipeline)

    groupers_description = description['analyze']['groupBy']
    str_groupers = ', '.join(_get_descriptions_name(groupers_description))
    module_logger.info(f"found groupers ({len(groupers_description)}): {str_groupers}")

    groupers = list(
        map(lambda cfg: create_instance(
            qualified_name=cfg['name'],
            kwargs=get_component_config(cfg),
            assert_base_classes=[AbstractGrouper]), groupers_description)
    )

    metrics_description = description['analyze']['metrics']
    str_metrics = ", ".join(_get_descriptions_name(metrics_description))
    module_logger.info(f"found metrics ({len(metrics_description)}): {str_metrics}")

    metrics = list(
        map(lambda cfg: create_instance(
            qualified_name=cfg['name'],
            kwargs=get_component_config(cfg),
            assert_base_classes=[AbstractAggregator]), metrics_description)
    )

    return AnalyzeWorkflowManager(
        description=description,
        data_adapter=source_adapter,
        pipeline_executor=pipeline_executor,
        groupers=groupers,
        metrics=metrics
    )