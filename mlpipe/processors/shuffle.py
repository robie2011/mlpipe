from dataclasses import dataclass

import numpy as np

from mlpipe.processors.interfaces import AbstractProcessor
from mlpipe.processors.standard_data_format import StandardDataFormat


@dataclass
class Shuffle(AbstractProcessor):
    def _process2d(self, processor_input: StandardDataFormat) -> StandardDataFormat:
        ix = np.arange(processor_input.data.shape[0])
        np.random.shuffle(ix)

        return processor_input.modify_copy(
            timestamps=processor_input.timestamps[ix],
            data=processor_input.data[ix])
