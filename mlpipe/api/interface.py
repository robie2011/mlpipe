from enum import Enum
from typing import List, Union, Dict

# contains special attributes:
#   name: qualified name
#   minutes: for aggregations
#   input/output fields:
#       - aggregator will get only releveant fields
#       - after aggregation we should name these fields correctly
from mlpipe.workflows.analyze.interface import AnalyzeDescription
from mlpipe.workflows.interface import ClassDescription

# class ClassConfig(TypedDict, Dict):
#     qualified_name: str
# Case 0: No configuration
# Case 1: Simple List: Use it as unnamed first argument
# Case 2: Dictionary -> kwargs
# config: Optional[Union[Dict, List]]

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


class PredictionTypes(Enum):
    BINARY = "binary"
    REGRESSION = "regression"
