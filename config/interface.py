from dataclasses import dataclass
from enum import Enum
from typing import List, Dict


@dataclass
class HistorySummary:
    epoch: List[int]
    params: Dict
    history: Dict


class TrainingProjectFileNames(Enum):
    MODEL = "model.h5"
    SCALERS = "scalers.pickle"
    DESCRIPTION = "description.json"
    HISTORY = "history.pickle"
    HISTORY_SUMMARY = "history_summary.pickle"