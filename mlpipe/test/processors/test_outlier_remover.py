import unittest
from typing import List
import numpy as np
from numpy.testing import assert_array_equal
import mlpipe.helpers.data as helper_data
from mlpipe.aggregators.outlier import InputOutputLimits
from mlpipe.processors.outlier_remover import OutlierRemover
from mlpipe.processors.standard_data_format import StandardDataFormat


class TestOutlierRemover(unittest.TestCase):
    def test_no_outliers(self):
        data = np.array([
            [10, 30],
            [12, 25]
        ], dtype="float64")
        data.flags.writeable = False

        input_data = StandardDataFormat(labels=['a', 'b'], data=data, timestamps=helper_data.generate_timestamps(2, 2))
        configs: List[InputOutputLimits] = [InputOutputLimits(inputField="a")]
        result = OutlierRemover(generate=configs)._process2d(input_data)
        assert_array_equal(data, result.data)

    def test_min_outlier(self):
        data = np.array([
            [10, 30],
            [12, 25]
        ], dtype="float64")
        data.flags.writeable = False

        input_data = StandardDataFormat(labels=['a', 'b'], data=data, timestamps=helper_data.generate_timestamps(2, 2))
        limits = [
            {'inputField': 'b', 'min': 28}
        ]

        result_expected = data.copy()
        result_expected[1, 1] = np.nan

        result = OutlierRemover(generate=limits)._process2d(input_data)
        assert_array_equal(result_expected, result.data)

    def test_max_outlier(self):
        data = np.array([
            [10, 30],
            [12, 25]
        ], dtype="float64")
        data.flags.writeable = False

        input_data = StandardDataFormat(labels=['a', 'b'], data=data, timestamps=helper_data.generate_timestamps(2, 2))
        limits = [
            {'inputField': 'a', 'max': 9},
        ]

        result_expected = data.copy()
        result_expected[:, 0] = np.nan

        result = OutlierRemover(generate=limits)._process2d(input_data)
        assert_array_equal(result_expected, result.data)

    def test_min_max_outlier(self):
        data = np.array([
            [10, 30],
            [12, 234]
        ], dtype="float64")
        data.flags.writeable = False

        input_data = StandardDataFormat(labels=['a', 'b'], data=data, timestamps=helper_data.generate_timestamps(2, 2))
        limits = [
            {'inputField': 'a', 'min': 11},
            {'inputField': 'b', 'max': 30}
        ]

        result_expected = data.copy()
        result_expected[0, 0] = np.nan
        result_expected[1, 1] = np.nan

        result = OutlierRemover(generate=limits)._process2d(input_data)
        assert_array_equal(result_expected, result.data)


if __name__ == '__main__':
    unittest.main()
