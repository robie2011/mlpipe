from typing import List, TypedDict, Union, Optional
from aggregators import AbstractAggregator
from api.interface import AnalyzeRequest, PipelineDescription, CreatePipelineRequest, CreateOrAnalyzePipeline
from api.class_loader import create_instance
from groupers import AbstractGrouper
from processors import AbstractProcessor
import logging
import yaml
from .pipeline_builder_interface import AnalyzerConfig, MultiAggregationConfig, Pipeline, SingleAggregationConfig, \
    BuildConfig

logger = logging.getLogger("pipeline.builder")


def get_component_config(key_values: dict):
    meta_config = ["name", "sequence", "generate"]
    return {k: v for (k, v) in key_values.items() if k not in meta_config}


def build(request: CreateOrAnalyzePipeline):
    logger.debug("build request received. structure: \n" + yaml.dump(request))
    source_definition = request['source']

    source = create_instance(
        source_definition['name'],
        get_component_config(source_definition))

    fields = request['sourceFields']

    pipeline: Optional[Pipeline] = None
    if 'pipeline' in request:
        pipeline = extract_pipeline(request['pipeline'])
        pipeline = reduce_pipeline(pipeline)

    analyzer: Optional[AnalyzerConfig] = None

    if 'analyze' in request:
        analyzer = extract_analyzer(request['analyze'])

    return BuildConfig(
        fields=fields,
        pipeline=pipeline,
        source=source,
        analyzer=analyzer
    )


def extract_analyzer(analyzer):
    group_by = list(
        map(lambda cfg: create_instance(
            qualified_name=cfg['name'],
            kwargs=get_component_config(cfg),
            assert_base_classes=[AbstractGrouper]), analyzer['groupBy'])
    )
    metrics = list(
        map(lambda cfg: create_instance(
            qualified_name=cfg['name'],
            kwargs=get_component_config(cfg),
            assert_base_classes=[AbstractAggregator]), analyzer['metrics'])
    )

    return AnalyzerConfig(group_by=group_by, aggregators=metrics)


def reduce_pipeline(pipeline: List[Union[AbstractProcessor, SingleAggregationConfig]]) -> Pipeline:
    reduced_pipeline: List[Union[MultiAggregationConfig, AbstractProcessor]] = []

    if isinstance(pipeline[0], SingleAggregationConfig):
        reduced_pipeline.append(MultiAggregationConfig(
            sequence=pipeline[0].sequence,
            instances=[pipeline[0]]
        ))
    else:
        reduced_pipeline.append(pipeline[0])

    for i in range(1, len(pipeline)):
        current_pipe = pipeline[i]

        if (isinstance(current_pipe, SingleAggregationConfig)
                and isinstance(reduced_pipeline[-1], MultiAggregationConfig)
                and reduced_pipeline[-1].sequence == current_pipe.sequence):
            reduced_pipeline[-1].instances.append(current_pipe)

        elif isinstance(current_pipe, SingleAggregationConfig):
            reduced_pipeline.append(
                MultiAggregationConfig(sequence=current_pipe.sequence, instances=[current_pipe])
            )

        elif isinstance(current_pipe, AbstractProcessor):
            # current_pipe is an abstract processor
            processor: AbstractProcessor = current_pipe
            reduced_pipeline.append(processor)

        else:
            raise Exception("Unsupported pipe (nr #{0}): {1}".format(i, current_pipe))

    logger.debug("reduced pipes: {0}".format(len(reduced_pipeline)))
    return reduced_pipeline


def extract_pipeline(descriptions: PipelineDescription):
    pipeline = []
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
