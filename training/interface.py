from dataclasses import dataclass
from typing import Optional, List, TypedDict

from workflows.interface import ClassDescription


class PreprocessingDescription(TypedDict):
    predictionSourceFields: List[str]
    predictionTargetField: str
    scale: Optional[List[ClassDescription]]
    create3dSequence: Optional[int]
    shuffle: Optional[bool]
    ratioTestdata: Optional[float]
