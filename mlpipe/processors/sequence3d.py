from dataclasses import dataclass
from typing import List
from . import StandardDataFormat
from .interfaces import AbstractProcessor
import numpy as np


def _create_sequence_endpoints(timestamps: np.ndarray, n_sequence: int) -> List[int]:
    # copied from p8 project
    """
    For given array of timestamps and required sequence length
    calculate valid sequence endpoints
    """

    # note: how to check for interruption
    #   if each following timestamp has only 1 minute difference
    #   than the time difference between timestamp on endpoint
    #   i and (i-5) should be 5 minutes
    #   therefore we can easily check whether sequence is interrupted or not
    n_past_values = n_sequence - 1

    output = np.full((timestamps.shape[0] - n_past_values, 2), np.nan, dtype='int')

    start_points = timestamps[:-n_past_values]
    end_points = timestamps[n_past_values:]
    deltas = end_points - start_points
    deltas_minute = deltas.astype('timedelta64[m]').astype('int')

    output[:, 0] = np.arange(n_past_values, timestamps.shape[0])
    output[:, 1] = deltas_minute

    # all valid sequence endpoints
    output_mask = output[:, 1] == n_past_values
    return output[output_mask][:, 0]


@dataclass
class Sequence3d(AbstractProcessor):
    sequence: int

    def process(self, processor_input: StandardDataFormat) -> StandardDataFormat:
        logger = self.get_logger()
        logger.info("create 3D-Sequence with sequence length={0}".format(self.sequence))
        ix_valid_endpoints = _create_sequence_endpoints(timestamps=processor_input.timestamps, n_sequence=self.sequence)
        logger.info("found {0:,} valid endpoints".format(ix_valid_endpoints.shape[0]))

        output_size = (ix_valid_endpoints.shape[0], self.sequence, processor_input.data.shape[1])
        output = np.full(output_size, np.nan)
        for i in range(len(ix_valid_endpoints)):
            ix_endpoint = ix_valid_endpoints[i]
            ix_end = ix_endpoint + 1 # because range end ix is exclusive
            ix_start = ix_end - self.sequence
            output[i] = processor_input.data[ix_start:ix_end]

        timestamps = processor_input.timestamps[ix_valid_endpoints]
        return processor_input.modify_copy(data=output, timestamps=timestamps)
