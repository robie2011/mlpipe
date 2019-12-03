from typing import List, Dict, TypedDict, Optional


# class ClassConfig(TypedDict, Dict):
#     qualified_name: str

    # Case 0: No configuration
    # Case 1: Simple List: Use it as unnamed first argument
    # Case 2: Dictionary -> kwargs
    #config: Optional[Union[Dict, List]]

# contains special attributes:
#   name: qualified name
#   minutes: for aggregations
#   input/output fields:
#       - aggregator will get only releveant fields
#       - after aggregation we should name these fields correctly
ClassDescription = Dict
PipelineDescription = List[ClassDescription]


class AnalyzeDescription(TypedDict):
    groupBy: List[str]
    metrics: List[ClassDescription]


class AnalyzeRequest(TypedDict):
    source: ClassDescription
    sourceFields: List[str]
    pipeline: Optional[PipelineDescription]
    analyze: AnalyzeDescription


class CreatePipeRequest(TypedDict):
    source: ClassDescription
    sourceFields: List[str]
    name: str
    pipeline: PipelineDescription


# TOOD: not finished
class ModelTrainRequest:
    pipelineName: str
    model: object  # todo: include generic modell
