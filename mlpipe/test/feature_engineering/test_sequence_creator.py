import unittest
from datetime import datetime, timedelta

import numpy as np
from numpy.testing import assert_array_equal

from mlpipe.test.helpers import print_3d_array
from mlpipe.processors.sequence3d import Sequence3d
from mlpipe.processors.standard_data_format import StandardDataFormat


def print_2darray(xs):
    for i in range(xs.shape[0]):
        print(xs[i, :])


data = np.array([np.arange(5), np.arange(50, 55)]).T

s = 3
n_sensors = 2
n_sequence = 3
n_length_new = data.shape[0] - n_sequence + 1


# noinspection PyMethodMayBeStatic
class SequenceCreatorTestCase(unittest.TestCase):
    def test_create_sequence(self):
        # test data:
        # https://docs.google.com/spreadsheets/d/1KoBUzJf4TIX5xlHIPg4BK6zDAugQWLJ7Lm_lOg2dcLg/edit#gid=589074770
        print_2darray(data)
        result_expected = np.zeros((n_length_new, n_sequence, n_sensors), dtype='int')

        result_expected[0, :, 0] = [0, 1, 2]
        result_expected[1, :, 0] = [1, 2, 3]
        result_expected[2, :, 0] = [2, 3, 4]

        result_expected[0, :, 1] = [50, 51, 52]
        result_expected[1, :, 1] = [51, 52, 53]
        result_expected[2, :, 1] = [52, 53, 54]

        result = Sequence3d.create_sequence_3d(features=data, n_sequence=3)
        print("result")
        print_2darray(result)
        assert_array_equal(result_expected, result)

    def test_create_sequence_timestamps_hole_before_last_entry(self):
        timedelta(minutes=1)

        stamps = np.arange(
            datetime(2019, 7, 1),
            datetime(2019, 7, 2),
            timedelta(minutes=1)).astype(datetime)[:5]
        stamps[-1] = stamps[-1] + timedelta(minutes=1)
        print(stamps)

        result_expected = np.zeros((n_length_new - 1, n_sequence, n_sensors))

        result_expected[0, :, 0] = [0, 1, 2]
        result_expected[1, :, 0] = [1, 2, 3]

        result_expected[0, :, 1] = [50, 51, 52]
        result_expected[1, :, 1] = [51, 52, 53]

        result = Sequence3d(sequence=3)._process2d(StandardDataFormat(timestamps=stamps, data=data, labels=['a', 'b']))
        print("result")
        print_3d_array(result.data)
        assert_array_equal(result_expected, result.data)


if __name__ == '__main__':
    unittest.main()
