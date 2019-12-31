import numpy as np
from mlpipe.processors import AbstractProcessor, StandardDataFormat
import logging


module_logger = logging.getLogger(__name__)


class NanRemover(AbstractProcessor):
    def process(self, processor_input: StandardDataFormat) -> StandardDataFormat:
        # index having zero nan in row
        ix_valid = np.sum(np.isnan(processor_input.data), axis=1) == 0

        return StandardDataFormat(
            labels=processor_input.labels,
            timestamps=processor_input.timestamps[ix_valid],
            data=processor_input.data[ix_valid]
        )

