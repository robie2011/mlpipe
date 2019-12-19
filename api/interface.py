from enum import Enum
from typing import List, Union, Dict

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
from workflows.analyzers.interface import AnalyzeDescription
from workflows.interface import ClassDescription

PipelineDescription = List[ClassDescription]

# python 3.8+
# class CreatePipelineRequest(TypedDict):
#     source: ClassDescription
#     sourceFields: List[str]
#     pipeline: PipelineDescription
CreatePipelineRequest = Dict


class AnalyzeRequest(CreatePipelineRequest):
    analyze: AnalyzeDescription


CreateOrAnalyzePipeline = Union[AnalyzeRequest, CreatePipelineRequest]


# TOOD: not finished
class ModelTrainRequest:
    pipelineName: str
    model: object  # todo: include generic modell


class PredictionTypes(Enum):
    BINARY = "BINARY"
    REGRESSION = "REGRESSION"
