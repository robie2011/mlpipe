from typing import List, Union
from aggregators import AbstractAggregator
from api.interface import PipelineDescription
from workflows.pipeline.interface import SingleAggregationConfig, MultiAggregation, \
    ProcessorOrMultiAggregation, PipelineWorkflow
from processors import AbstractProcessor
from workflows.utils import get_component_config, create_instance
import logging

logger = logging.getLogger(__name__)

ProcessorOrAggregation = Union[AbstractProcessor, SingleAggregationConfig]


def _initialize_processors_and_aggregators(descriptions: PipelineDescription):
    pipeline: List[ProcessorOrAggregation] = []
    for definition in descriptions:
        config = get_component_config(definition)
        if 'sequence' in definition:
            single_aggregation = SingleAggregationConfig(
                sequence=definition['sequence'],
                generate=definition['generate'],
                instance=create_instance(
                    qualified_name=definition['name'],
                    kwargs=config,
                    assert_base_classes=[AbstractAggregator])
            )

            pipeline.append(single_aggregation)

        else:
            config = get_component_config(definition)
            instance = create_instance(
                qualified_name=definition['name'],
                kwargs=config,
                assert_base_classes=[AbstractProcessor])
            pipeline.append(instance)

    logger.debug("extracted pipes: {0}".format(len(pipeline)))
    return pipeline


def _reduce_pipeline(pipeline: List[ProcessorOrAggregation]) -> List[ProcessorOrMultiAggregation]:
    reduced_pipeline: List[ProcessorOrMultiAggregation] = []

    if isinstance(pipeline[0], SingleAggregationConfig):
        reduced_pipeline.append(MultiAggregation(
            sequence=pipeline[0].sequence,
            instances=[pipeline[0]]
        ))
    else:
        reduced_pipeline.append(pipeline[0])

    for i in range(1, len(pipeline)):
        current_pipe = pipeline[i]

        if (isinstance(current_pipe, SingleAggregationConfig)
                and isinstance(reduced_pipeline[-1], MultiAggregation)
                and reduced_pipeline[-1].sequence == current_pipe.sequence):
            reduced_pipeline[-1].instances.append(current_pipe)

        elif isinstance(current_pipe, SingleAggregationConfig):
            reduced_pipeline.append(
                MultiAggregation(sequence=current_pipe.sequence, instances=[current_pipe])
            )

        elif isinstance(current_pipe, AbstractProcessor):
            # current_pipe is an abstract processor
            processor: AbstractProcessor = current_pipe
            reduced_pipeline.append(processor)

        else:
            raise Exception("Unsupported pipe (nr #{0}): {1}".format(i, current_pipe))

    logger.debug("reduced pipes: {0}".format(len(reduced_pipeline)))
    return reduced_pipeline


def create_pipeline_workflow(descriptions: PipelineDescription) -> PipelineWorkflow:
    flatten_pipeline = _initialize_processors_and_aggregators(descriptions)
    reduced_piplines = _reduce_pipeline(flatten_pipeline)
    return PipelineWorkflow(pipelines=reduced_piplines)
