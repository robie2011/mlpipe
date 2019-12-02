from dataclasses import dataclass
from typing import List, TypedDict, Union
from aggregators import AbstractAggregator
from api.api_desc import AnalyzeRequest
from api.class_loader import create_instance
from processors import AbstractProcessor

metaConfig = ["name", "minutes", "generate"]


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
    return 'minutes' in definition


def build(request: AnalyzeRequest):
    source_definition = request['source']

    source: EmpaCsvReader = create_instance(
        source_definition['name'],
        _get_config(source_definition))

    pipeline = _reduce_pipeline(_extract_pipeline(request))

    return source, pipeline


def _reduce_pipeline(pipeline: List[Union[AbstractProcessor, SingleAggregation]]) -> Pipeline:
    reduced_pipeline = []

    if isinstance(pipeline[0], SingleAggregation):
        reduced_pipeline.append(MultiAggregationConfig(
            minutes=pipeline[0].sequence,
            instances=[pipeline[0]]
        ))
    else:
        reduced_pipeline.append(pipeline[0])

    for i in range(1, len(pipeline)):
        current_pipe = pipeline[i]
        is_aggregation = isinstance(current_pipe, SingleAggregation)
        if (is_aggregation
                and isinstance(reduced_pipeline[-1], MultiAggregationConfig)
                and reduced_pipeline[-1].n_sequence == current_pipe.sequence):
            reduced_pipeline[-1].instances.append(current_pipe)
        elif is_aggregation:
            reduced_pipeline.append(
                MultiAggregationConfig(minutes=current_pipe.sequence, instances=[current_pipe])
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
