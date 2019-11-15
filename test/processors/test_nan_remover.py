import unittest
import numpy as np
from numpy.testing import assert_array_equal
import helpers.data as helper_data
from processors import *
from processors.nan_remover import NanRemover


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
        process_data = ProcessorData(data=data, labels=['a', 'b'], timestamps=helper_data.generate_timestamps(2, samples=4))
        NanRemover().process(process_data)


if __name__ == '__main__':
    unittest.main()
