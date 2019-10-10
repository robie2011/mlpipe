import unittest
import numpy as np

from datasources import DataResult
from preprocessors import NanRemover
from numpy.testing import assert_array_equal


class TestNanProcessor(unittest.TestCase):
    def test_filtering(self):
        data = np.array([
            [1, 2, 3],
            [4, 5, np.nan],
            [7, 8, 9],
            [np.nan, np.nan, np.nan]])

        stamps = np.arange(
            np.datetime64('2019-04-19'),
            np.datetime64('2019-04-20'),
            np.timedelta64(1, 'h'))
        processorInput = DataResult(values=data, timestamps=stamps[:4], columns=['A', 'B', 'C'])

        processor = NanRemover()
        result = processor.process(processorInput)

        self.assertEqual(result.columns, ['A', 'B', 'C'])
        assert_array_equal(result.values, np.array([
            [1, 2, 3],
            [7, 8, 9],
        ]))


if __name__ == '__main__':
    unittest.main()
