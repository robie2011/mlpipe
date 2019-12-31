import unittest
from mlpipe.encoders.one_hot_encoder import OneHotEncoder
import numpy as np
from numpy.testing import assert_array_equal

from mlpipe.helpers import transform_to_2d_matrix


class TestOneHotEncoding(unittest.TestCase):
    def test_one_hot_encoding_number(self):
        encoding = transform_to_2d_matrix(np.array([0, 7, 23, 7, 23]))
        encoder = OneHotEncoder(encoding)

        data = transform_to_2d_matrix(np.array([0, 23, 7, 7]))
        result_expected = np.zeros((4, 3))
        result_expected[0, 0] = 1
        result_expected[1, 2] = 1
        result_expected[2, 1] = 1
        result_expected[3, 1] = 1

        result = encoder.encode(data_1d=data)
        assert_array_equal(result_expected, result)

    def test_one_hot_encoding_string(self):
        encoding = transform_to_2d_matrix(np.array(["Auto", "Velo", "Bus"]))
        encoder = OneHotEncoder(encoding)

        data = transform_to_2d_matrix(np.array(["Auto", "Velo", "Bus", "Bus"]))
        result_expected = np.zeros((4, 3))
        result_expected[0, 0] = 1
        result_expected[1, 2] = 1
        result_expected[2, 1] = 1
        result_expected[3, 1] = 1

        result = encoder.encode(data_1d=data)
        assert_array_equal(result_expected, result)


if __name__ == '__main__':
    unittest.main()
