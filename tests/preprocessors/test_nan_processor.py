import unittest
import numpy as np
from preprocessors import NanProcessor
from numpy.testing import assert_array_equal


class TestNanProcessor(unittest.TestCase):
    def test_filtering(self):
        data = np.array([
            [1, 2, 3],
            [4, 5, np.nan],
            [7, 8, 9],
            [np.nan, np.nan, np.nan]])

        processor = NanProcessor()
        result = processor.process(data, ['A', 'B', 'C'])

        self.assertEqual(result.columns, ['A', 'B', 'C'])
        assert_array_equal(result.values, np.array([
            [1, 2, 3],
            [7, 8, 9],
        ]))


if __name__ == '__main__':
    unittest.main()
