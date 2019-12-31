from dataclasses import dataclass


@dataclass
class ModelLocation:
  name: str
  session_id: str
  path: str
  sizeBytes: int
  monitored_value: float
  accuracy: float
  epochs: int
  batch_size: int
  samples: int
  metrics: str
  datetime: str