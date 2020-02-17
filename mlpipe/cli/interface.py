from dataclasses import dataclass
from datetime import datetime


@dataclass
class ModelLocation:
    name: str
    session_id: str
    path: str
    sizeBytes: int
    epochs: int
    batch_size: int
    samples: int
    metrics: str
    datetime: datetime
