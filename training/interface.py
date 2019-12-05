from dataclasses import dataclass
from typing import Optional, List
from api.interface import ClassDescription


class PreprocessingDescription(TypedDict):
    dropFields: Optional[List[str]]
    scale: Optional[List[ClassDescription]]
    create3dSequence: Optional[int]
    shuffle: Optional[bool]
    ratioTestdata: Optional[float]
