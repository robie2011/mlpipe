import numpy as np
from processors import AbstractProcessor, ProcessorData


class NanRemover(AbstractProcessor):
    def process(self, processor_input: ProcessorData) -> ProcessorData:
        index_has_nans = np.sum(np.isnan(processor_input.data), axis=1) > 0
        return ProcessorData(
            labels=processor_input.labels,
            timestamps=processor_input.timestamps[index_has_nans],
            data=processor_input.data[index_has_nans]
        )

