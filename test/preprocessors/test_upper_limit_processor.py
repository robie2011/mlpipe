import unittest
import numpy as np
from preprocessors import RangeLimiter
from numpy.testing import assert_array_equal
from datasources import DataResult


class TestUpperLimitProcessor(unittest.TestCase):
    def test_no_limit(self):
        processor = RangeLimiter(limits=[1000, 1000, 1000])
        stamps = np.arange(
            np.datetime64('2019-04-19'),
            np.datetime64('2019-04-20'),
            np.timedelta64(1, 'h'))

        data = DataResult(values=np.array([
            [1, 30, 50],
            [2, 31, 51],
            [3, 32, 52]]),
            timestamps=stamps[:3],
            columns=['A', 'B', 'C'])

        result = processor.process(data)

        assert_array_equal(data.columns, result.columns)
        assert_array_equal(data.timestamps, result.timestamps)
        assert_array_equal(data.values, result.values)

    def test_limit_first_column(self):
        processor = RangeLimiter(limits=[11, 1000, 1000])
        stamps = np.arange(
            np.datetime64('2019-04-19'),
            np.datetime64('2019-04-20'),
            np.timedelta64(1, 'h'))

        data = DataResult(values=np.array([
            [10, 30, 50],
            [13, 31, 51],
            [15, 32, 52]]),
            timestamps=stamps[:3],
            columns=['A', 'B', 'C'])

        result = processor.process(data)
        assert_array_equal(data.columns, result.columns)
        assert_array_equal(data.timestamps[:1], result.timestamps)
        assert_array_equal(data.values[:1], result.values)

    def test_limit_multiple_column(self):
        processor = RangeLimiter(limits=[14, 31, 1000])
        stamps = np.arange(
            np.datetime64('2019-04-19'),
            np.datetime64('2019-04-20'),
            np.timedelta64(1, 'h'))

        data = DataResult(values=np.array([
            [10, 30, 50],
            [13, 31, 51],
            [15, 32, 52]]),
            timestamps=stamps[:3],
            columns=['A', 'B', 'C'])

        result = processor.process(data)
        assert_array_equal(data.columns, result.columns)
        assert_array_equal(data.timestamps[:2], result.timestamps)
        assert_array_equal(data.values[:2], result.values)


if __name__ == '__main__':
    unittest.main()
