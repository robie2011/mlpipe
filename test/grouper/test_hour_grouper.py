import unittest
import numpy as np
from numpy.testing import assert_array_equal
from helpers.data import print_3d_array
from groupers import create_sequence_offset_matrix
from datetime import datetime, timedelta
from groupers import GroupInput, HourGrouper


class TestHourGrouper(unittest.TestCase):
    def test_hour_grouper(self):
        raw_data = np.zeros((20, 3), dtype='object')
        raw_data[:, 1:] = np.array([
            [23.0, 10],
            [23.0, 10],
            [23.3, 11],
            [23.1, 10],

            [23.0, 11],
            [23.0, 12],
            [23.0, 13],
            [23.0, 14],

            [23.0, 14],
            [24.0, 14],
            [25.0, 14],
            [25.0, 14],

            [24.0, 14],
            [25.0, 14],
            [26.0, 14],
            [26.0, 14],

            [27.0, 14],
            [27.0, 14],
            [27.0, 14],
            [28.0, 14]
        ])

        _stamps = np.arange(datetime(2019, 7, 1), datetime(2019, 7, 2), timedelta(minutes=15)).astype(datetime)
        raw_data[:, 0] = _stamps[:raw_data.shape[0]]
        result = HourGrouper().group(GroupInput(raw_data=raw_data, timestamps=raw_data[:,0]))
        result_expected = np.array([
            0,  # :00
            0,  # :15
            0,  # :30
            0,  # :45

            1,
            1,
            1,
            1,

            2,
            2,
            2,
            2,

            3,
            3,
            3,
            3,

            4,
            4,
            4,
            4
        ])

        assert_array_equal(result_expected, result)


