from mlpipe.aggregators import AbstractAggregator
from mlpipe.groupers import AbstractGrouper
from .interface import AnalyzeDescription
from .analyzer_workflow import AnalyzerWorkflow
from mlpipe.workflows.utils import get_component_config, create_instance


def create_analyzer_workflow(desc_analyze: AnalyzeDescription) -> AnalyzerWorkflow:
    group_by = list(
        map(lambda cfg: create_instance(
            qualified_name=cfg['name'],
            kwargs=get_component_config(cfg),
            assert_base_classes=[AbstractGrouper]), desc_analyze['groupBy'])
    )
    metrics = list(
        map(lambda cfg: create_instance(
            qualified_name=cfg['name'],
            kwargs=get_component_config(cfg),
            assert_base_classes=[AbstractAggregator]), desc_analyze['metrics'])
    )

    return AnalyzerWorkflow(group_by=group_by, aggregators=metrics)
