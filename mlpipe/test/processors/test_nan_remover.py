import unittest

import numpy as np
from numpy.testing import assert_array_equal

import mlpipe.test.helpers.data as helper_data
from mlpipe.processors.nan_remover import NanRemover
from mlpipe.processors.standard_data_format import StandardDataFormat


class TestNanRemover(unittest.TestCase):
    def test_standard_case(self):
        data = np.array([
            [23, 55],
            [21, 52],
            [np.nan, 52],
            [23, np.nan],
        ])
        data.flags.writeable = False

        result_excepted = data[:2]
        process_data = StandardDataFormat(
            data=data,
            labels=['a', 'b'],
            timestamps=helper_data.generate_timestamps(2, samples=4))
        result = NanRemover()._process2d(process_data)
        assert_array_equal(result_excepted, result.data)


if __name__ == '__main__':
    unittest.main()
