import helpers.data as helper_data
import unittest
from processors import *
import numpy as np
from datetime import datetime, timedelta
from numpy.testing import assert_array_equal

from processors.freezed_value_remover import FreezedValueRemover


class TestFreezedValueRemover(unittest.TestCase):
    def test_standard_case(self):
        data = np.array([
            [11, 20],
            [12, 21],
            [13, 22],
            [14, 20],
            [15, 20],
            [16, 20],
            [17, 20],
            [18, 20]
        ], dtype='float')

        result_expected = np.array([
            [11, 20],
            [12, 21],
            [13, 22],
            [14, 20],
            [15, 20],
            [16, 20],
            [17, np.nan],
            [18, np.nan]
        ], dtype='float')
        processor_data = ProcessorData(
            timestamps=helper_data.generate_timestamps(samples=data.shape[0]),
            labels=helper_data.get_labels(2),
            data=data
        )

        processor_data_result = FreezedValueRemover(max_freezed_values=3).process(processor_data)
