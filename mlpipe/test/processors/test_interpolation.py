import unittest

import numpy as np
from numpy.testing import assert_array_equal

from mlpipe.test.helpers import data as helper_data
from mlpipe.processors.interpolation import Interpolation
from mlpipe.processors.standard_data_format import StandardDataFormat


class TestInterpolation(unittest.TestCase):
    def test_standardcase(self):
        data = np.array([
            [13, 20],
            [14, 21],
            [np.nan, 22],
            [16, np.nan],
            [17, np.nan],
            [np.nan, np.nan],
            [19, np.nan],
            [20, 27],
        ], dtype="float")
        data.flags.writeable = False

        result_expected = np.array([
            [13, 20],
            [14, 21],
            [15, 22],
            [16, 23],
            [17, 24],
            [18, 25],
            [19, np.nan],
            [20, 27],
        ], dtype="float")
        result_expected.flags.writeable = False

        result = Interpolation(max_consecutive_interpolated_value=3)._process2d(
            StandardDataFormat(labels=['a', 'b'], data=data, timestamps=helper_data.generate_timestamps(samples=8))
        )
        assert_array_equal(result_expected, result.data)


if __name__ == '__main__':
    unittest.main()
