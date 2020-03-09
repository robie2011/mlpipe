import logging
from typing import List, cast

from mlpipe.aggregators.abstract_aggregator import AbstractAggregator
from mlpipe.api.interface import PipelineDescription
from mlpipe.dsl_interpreter.descriptions import ExecutionModes
from mlpipe.exceptions.interface import MLPipeError
from mlpipe.processors.interfaces import AbstractProcessor
from mlpipe.processors.internal.multi_aggregation import MultiAggregation
from mlpipe.pipeline.pipeline_executor import PipelineExecutor
from mlpipe.workflows.utils import get_component_config, create_instance

module_logger = logging.getLogger(__name__)


def _initialize_processors_and_aggregators(descriptions: PipelineDescription):
    pipeline: List[AbstractProcessor] = []
    for definition in descriptions:
        try:
            config = get_component_config(definition)
            instance: AbstractProcessor = create_instance(
                qualified_name=definition['name'],
                kwargs=config,
                assert_base_classes=[AbstractProcessor])
            instance.id = definition['@id']

            pipeline.append(instance)
        except MLPipeError as e:
            module_logger.error(f"can not initialize pipe: {definition.get('name', 'NAME UNDEFINED')}")
            module_logger.error(e)
            raise

    module_logger.debug("extracted filters: {0}".format(len(pipeline)))
    return pipeline


def _reduce_pipeline(pipeline: List[AbstractProcessor]) -> List[AbstractProcessor]:
    reduced_pipeline: List[AbstractProcessor] = []
    if not pipeline:
        return reduced_pipeline

    for ix, pipe in enumerate(pipeline):
        # case 1: invalid instance
        if not isinstance(pipe, AbstractProcessor):
            raise Exception("Unsupported pipe (nr #{0}): {1}".format(ix, pipe))

        # case 2: normal processor
        if not isinstance(pipe, AbstractAggregator):
            reduced_pipeline.append(pipe)
            continue

        # case aggregators:
        # case 3a: reduced pipe is empty or last reduced pipe is not multiaggregation
        agg = cast(AbstractAggregator, pipe)
        if not isinstance(agg, AbstractAggregator):
            raise Exception("Should not happen here")

        if not reduced_pipeline \
                or not isinstance(reduced_pipeline[-1], MultiAggregation) \
                or cast(MultiAggregation, reduced_pipeline[-1]).sequence != agg.sequence:
            reduced_pipeline.append(MultiAggregation.from_aggregator(aggregator=agg))
            continue

        # case 3b:
        # - reduced pipe is not empty
        # - last instance is multi aggregation
        # - multi aggregation has the same sequence length
        cast(MultiAggregation, reduced_pipeline[-1]).instances.append(agg)

    module_logger.debug("reduced filters: {0}".format(len(reduced_pipeline)))
    return reduced_pipeline


def _filter_execution_mode(descriptions: PipelineDescription, execution_mode: ExecutionModes) -> PipelineDescription:
    description_filtered = []
    for d in descriptions:
        if '_condition' not in d or d['_condition'] == execution_mode.name:
            description_filtered.append(d)
        else:
            module_logger.info(f"droping pipe {d['name']} because @mode={d['_condition']} required")
    return description_filtered


def build_pipeline_executor(descriptions: PipelineDescription, execution_mode: ExecutionModes) -> PipelineExecutor:
    set_pipe_identification(descriptions)
    description_filtered = _filter_execution_mode(descriptions, execution_mode)
    flatten_pipeline = _initialize_processors_and_aggregators(description_filtered)
    reduced_pipline = _reduce_pipeline(flatten_pipeline)
    return PipelineExecutor(pipeline=reduced_pipline)


def set_pipe_identification(descriptions):
    for ix, d in enumerate(descriptions):
        d['@id'] = str(ix)
