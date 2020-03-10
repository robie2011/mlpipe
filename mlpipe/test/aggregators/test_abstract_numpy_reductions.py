import unittest

import numpy as np
from numpy.testing import assert_array_equal

from mlpipe.aggregators.counter import Counter
from mlpipe.aggregators.max import Max
from mlpipe.aggregators.mean import Mean
from mlpipe.aggregators.min import Min
from mlpipe.aggregators.nan_counter import NanCounter
from mlpipe.aggregators.sum import Sum

sequences = np.array([
    [23.0, 10],
    [23.0, 10],
    [23.3, 11],
    [23.1, 10],
    [23.0, 11],
    [23.0, np.nan],
    [23.0, np.nan],
    [23.0, 14],
    [23.0, 14],
    [24.0, 14],
    [15.0, 14],  # modified
    [35.0, 14],  # modified
    [24.0, 14],
    [np.nan, 14],
    [26.0, 14],
    [26.0, 14],
    [26.0, 14],
    [26.0, 14],
    [27.0, 14],
    [28.0, 14]
])

sequences.flags.writeable = False
seq_3d = np.expand_dims(sequences, axis=0)


class TestAbstractNumpyReductions(unittest.TestCase):
    def test_counter_aggregate(self):
        result = Counter(sequence=np.nan, generate=[]).aggregate(
            grouped_data=seq_3d
        )

        result_a, result_b = result[0]

        self.assertEqual(result_a, sequences.shape[0])
        self.assertEqual(result_b, sequences.shape[0])

    def test_max_aggregate(self):
        result = Max(sequence=np.nan, generate=[]).aggregate(
            grouped_data=seq_3d
        )

        result_a, result_b = result[0]
        self.assertEqual(result_a, 35)
        self.assertEqual(result_b, 14)

    def test_min_aggregate(self):
        result = Min(sequence=np.nan, generate=[]).aggregate(
            grouped_data=seq_3d
        )

        result_a, result_b = result[0]
        self.assertEqual(result_a, 15)
        self.assertEqual(result_b, 10)

    def test_mean_aggregate(self):
        result = Mean(sequence=np.nan, generate=[]).aggregate(
            grouped_data=seq_3d
        )

        result_a, result_b = result[0]
        self.assertEqual(result_a, 24.442105263157895)
        self.assertEqual(result_b, 13)

    def test_nan_counter_aggregate(self):
        result = NanCounter(sequence=np.nan, generate=[]).aggregate(
            grouped_data=seq_3d
        )

        result_a, result_b = result[0]
        self.assertEqual(result_a, 1)
        self.assertEqual(result_b, 2)

    def test_sum_aggregate(self):
        result = Sum(sequence=np.nan, generate=[]).aggregate(
            grouped_data=seq_3d
        )

        result_a, result_b = result[0]
        self.assertEqual(result_a, 464.4)
        self.assertEqual(result_b, 234)


if __name__ == '__main__':
    unittest.main()
