from dataclasses import dataclass
from typing import List, Tuple, TypedDict, Union, Optional

from aggregators import AbstractAggregator
from api.api_desc import AnalyzeRequest
from api.class_loader import load, create_instance
from datasources.empa import EmpaCsvReader
from processors import AbstractProcessor

metaConfig = ["name", "minutes", "generate"]

class InputOutputField(TypedDict):
    inputField: str
    outputField: str


@dataclass
class SingleAggregation:
    minutes: int
    instance: AbstractAggregator
    generate: List[InputOutputField]


@dataclass
class MultiAggregation:
    minutes: int
    instances: List[SingleAggregation]


Pipeline = List[Union[AbstractProcessor, MultiAggregation]]


def _get_config(key_values: dict):
    return {k: v for (k, v) in key_values.items() if k not in metaConfig}


def _is_definition_for_aggregation(definition: dict):
    return 'minutes' in definition


def build(request: AnalyzeRequest):
    source_definition = request['source']

    source: EmpaCsvReader = create_instance(
        source_definition['name'],
        _get_config(source_definition))

    pipeline = _reduce_pipeline(_extract_pipeline(request))

    return pipeline


def _reduce_pipeline(pipeline: List[Union[AbstractProcessor, SingleAggregation]]) -> Pipeline:
    reduced_pipeline = []

    if isinstance(pipeline[0], SingleAggregation):
        reduced_pipeline.append(MultiAggregation(
            minutes=pipeline[0].minutes,
            instances=[pipeline[0]]
        ))
    else:
        reduced_pipeline.append(pipeline[0])

    for i in range(1, len(pipeline)):
        current_pipe = pipeline[i]
        is_aggregation = isinstance(current_pipe, SingleAggregation)
        if (is_aggregation
                and isinstance(reduced_pipeline[-1], MultiAggregation)
                and reduced_pipeline[-1].minutes == current_pipe.minutes):
            reduced_pipeline[-1].instances.append(current_pipe)
        elif is_aggregation:
            reduced_pipeline.append(
                MultiAggregation(minutes=current_pipe.minutes, instances=[current_pipe])
            )
        else:
            reduced_pipeline.append(current_pipe)
    return reduced_pipeline


def _extract_pipeline(request):
    pipeline = []
    for definition in request['pipeline']:
        config = _get_config(definition)
        if _is_definition_for_aggregation(definition):
            single_aggregation = SingleAggregation(
                minutes=definition['minutes'],
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
    return pipeline