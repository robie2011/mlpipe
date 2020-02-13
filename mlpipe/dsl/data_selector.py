from dataclasses import dataclass
from typing import List


@dataclass
class InputDataSelector:
    input_labels: List[str]

    def select(self):
        pass

