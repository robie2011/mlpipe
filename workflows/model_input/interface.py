from typing import TypedDict, List, Optional, Collection
from workflows.interface import ClassDescription


class PreprocessingDescription(TypedDict):
    predictionSourceFields: List[str]
    predictionTargetField: str
    scale: Optional[List[ClassDescription]]
    create3dSequence: Optional[int]
    shuffle: Optional[bool]
    ratioTestdata: Optional[float]
    dropFields: Optional[List[str]]

