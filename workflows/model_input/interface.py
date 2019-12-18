from typing import List, Optional, Collection, Dict
from workflows.interface import ClassDescription


# python 3.8+
# class PreprocessingDescription(TypedDict):
#     predictionSourceFields: List[str]
#     predictionTargetField: str
#     scale: Optional[List[ClassDescription]]
#     create3dSequence: Optional[int]
#     shuffle: Optional[bool]
#     ratioTestdata: Optional[float]
#     dropFields: Optional[List[str]]
PreprocessingDescription = Dict

