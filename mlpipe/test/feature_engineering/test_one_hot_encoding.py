import unittest

import numpy as np
from numpy.testing import assert_array_equal

from mlpipe.test.helpers import transform_to_2d_matrix, generate_timestamps
from mlpipe.processors.range_encoder import RangeEncoder
from mlpipe.processors.standard_data_format import StandardDataFormat


class TestOneHotEncoding(unittest.TestCase):
    def test_range_encoder(self):
        generates = [
            {'inputField': 'hour', 'outputField': 'hourOneHot'}
        ]

        encoder = RangeEncoder(generate=generates, value_from=0, value_to=3)

        data = transform_to_2d_matrix(np.array([0, 1, 1, 2]))
        result_expected = np.zeros((4, 3))
        result_expected[0, 0] = 1
        result_expected[1, 1] = 1
        result_expected[2, 1] = 1
        result_expected[3, 2] = 1

        result = encoder._process2d(StandardDataFormat(
            data=data, labels=['hour'], timestamps=generate_timestamps(samples=result_expected.shape[0])))
        assert_array_equal(result_expected, result.data)

    def test_range_encoder_two_cols(self):
        generates = [
            {'inputField': 'hour', 'outputField': 'hourOneHot'}
        ]

        encoder = RangeEncoder(generate=generates, value_from=0, value_to=3)

        data = np.array([
            [0, 1, 1, 2],
            [1, 2, 3, 4]
        ]).T

        result_expected = np.zeros((4, 4))
        result_expected[0, 1 + 0] = 1
        result_expected[1, 1 + 1] = 1
        result_expected[2, 1 + 1] = 1
        result_expected[3, 1 + 2] = 1
        result_expected[:, 0] = np.array([1, 2, 3, 4])

        result = encoder._process2d(StandardDataFormat(
            data=data, labels=['hour', 'abc'], timestamps=generate_timestamps(samples=result_expected.shape[0])))
        assert_array_equal(result_expected, result.data)
        self.assertEqual("abc", result.labels[0])
        self.assertEqual("hourOneHot$0", result.labels[1])
        self.assertEqual("hourOneHot$1", result.labels[2])
        self.assertEqual("hourOneHot$2", result.labels[3])


if __name__ == '__main__':
    unittest.main()
