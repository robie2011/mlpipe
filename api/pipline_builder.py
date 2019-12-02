from dataclasses import dataclass
from typing import List, TypedDict, Union
from aggregators import AbstractAggregator
from api.api_desc import AnalyzeRequest
from api.class_loader import create_instance
from processors import AbstractProcessor
import logging
import yaml


logger = logging.getLogger("pipeline.builder")

metaConfig = ["name", "sequence", "generate"]


class InputOutputField(TypedDict):
    inputField: str
    outputField: str


@dataclass
class SingleAggregation:
    sequence: int
    instance: AbstractAggregator
    generate: List[InputOutputField]


@dataclass
class MultiAggregationConfig:
    sequence: int
    instances: List[SingleAggregation]


Pipeline = List[Union[AbstractProcessor, MultiAggregationConfig]]


def _get_config(key_values: dict):
    return {k: v for (k, v) in key_values.items() if k not in metaConfig}


def _is_definition_for_aggregation(definition: dict):
    return 'sequence' in definition


def build(request: AnalyzeRequest):
    logger.debug("build request received. structure: \n" + yaml.dump(request))
    source_definition = request['source']

    source = create_instance(
        source_definition['name'],
        _get_config(source_definition))

    fields = request['sourceFields']

    pipeline = _reduce_pipeline(
        _extract_pipeline(request))

    return source, fields, pipeline


def _reduce_pipeline(pipeline: List[Union[AbstractProcessor, SingleAggregation]]) -> Pipeline:
    reduced_pipeline: List[Union[MultiAggregationConfig, AbstractProcessor]] = []
    current_pipe = pipeline[0]

    if isinstance(pipeline[0], SingleAggregation):
        reduced_pipeline.append(MultiAggregationConfig(
            sequence=pipeline[0].sequence,
            instances=[pipeline[0]]
        ))
    else:
        reduced_pipeline.append(pipeline[0])

    for i in range(1, len(pipeline)):
        current_pipe = pipeline[i]

        if (isinstance(current_pipe, SingleAggregation)
                and isinstance(reduced_pipeline[-1], MultiAggregationConfig)
                and reduced_pipeline[-1].sequence == current_pipe.sequence):
            reduced_pipeline[-1].instances.append(current_pipe)

        elif isinstance(current_pipe, SingleAggregation):
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


def _extract_pipeline(request):
    pipeline = []
    for definition in request['pipeline']:
        config = _get_config(definition)
        if _is_definition_for_aggregation(definition):
            single_aggregation = SingleAggregation(
                sequence=definition['sequence'],
                generate=definition['generate'],
                instance=create_instance(
                    qualified_name=definition['name'],
                    kwargs=config,
                    assert_base_classes=[])
            )

            pipeline.append(single_aggregation)

        else:
            config = _get_config(definition)
            instance = create_instance(
                qualified_name=definition['name'],
                kwargs=config,
                assert_base_classes=[AbstractProcessor])
            pipeline.append(instance)

    logger.debug("extracted pipes: {0}".format(len(pipeline)))
    return pipeline
