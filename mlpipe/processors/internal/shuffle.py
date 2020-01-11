from dataclasses import dataclass
from mlpipe.processors import AbstractProcessor, StandardDataFormat
import numpy as np


@dataclass
class Shuffle(AbstractProcessor):
    def process(self, processor_input: StandardDataFormat) -> StandardDataFormat:
        ix = np.arange(processor_input.data.shape[0])
        np.random.shuffle(ix)

        return processor_input.modify_copy(
            timestamps=processor_input.timestamps[ix],
            data=processor_input.data[ix])
