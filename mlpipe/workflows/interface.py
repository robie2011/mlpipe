from typing import Optional

from typing_extensions import TypedDict


class ClassDescription(TypedDict):
    name: str
    _condition: Optional[str]

