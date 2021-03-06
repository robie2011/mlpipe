import unittest

import numpy as np
from numpy.testing import assert_array_equal

import mlpipe.test.helpers.data as helper_data
from mlpipe.aggregators.max import Max
from mlpipe.aggregators.mean import Mean
from mlpipe.aggregators.min import Min
from mlpipe.aggregators.nan_counter import NanCounter
from mlpipe.aggregators.outlier import Outlier
from mlpipe.aggregators.percentile import Percentile
# data = helper_data.generated_3d_data()
# helpers.print_3d_array
from mlpipe.aggregators.sum import Sum
from mlpipe.aggregators.trend import Trend

"""
    sensor  0
    t=0  [417. 147. 397. 204. 417.]
    t=1  [801. 876. 170. 958. 687.]
    t=2  [989. 103. 288. 212. 574.]

    sensor  1
    t=0  [720.  92. 539. 878. 559.]
    t=1  [968. 895. 878. 533. 835.]
    t=2  [748. 448. 130. 266. 147.]

    sensor  2
    t=0  [  0. 186. 419.  27. 140.]
    t=1  [313.  85.  98. 692.  18.]
    t=2  [280. 909.  19. 492. 589.]

    sensor  3
    t=0  [302. 346. 685. 670. 198.]
    t=1  [692.  39. 421. 316. 750.]
    t=2  [789. 294. 679.  53. 700.]
"""


class TestSimpleAggregators(unittest.TestCase):
    def test_nan_counter(self):
        data = helper_data.generated_3d_data()
        data[0, 0, 0] = np.nan  # first sensor has 1 nan in measurement nr. 0
        data[1, 0:2, 1] = np.nan  # second sensor has 2 nans measurement nr. 1
        data[0:2, 0:2, 2] = np.nan  # third sensor has 2 nans in measurement nr. 0 and in measurement nr.1

        result_expected = np.zeros((3, 4))
        result_expected[0, 0] = 1
        result_expected[1, 1] = 2
        result_expected[0:2, 2] = 2

        result = NanCounter(sequence=np.nan).aggregate(grouped_data=data)
        assert_array_equal(result_expected, result)

    def test_max(self):
        result_expected = np.array([
            [417, 878, 419, 685],
            [958, 968, 692, 750],
            [989, 748, 909, 789]
        ])

        result = Max(sequence=np.nan).aggregate(grouped_data=helper_data.generated_3d_data())
        assert_array_equal(result_expected, result)

    def test_min(self):
        result_expected = np.array([
            147, 92, 0, 198,
            170, 533, 18, 39,
            103, 130, 19, 53]).reshape(3, 4)

        result = Min(sequence=np.nan).aggregate(grouped_data=helper_data.generated_3d_data())
        assert_array_equal(result_expected, result)

    def test_mean(self):
        grouped_data = helper_data.generated_3d_data()
        helper_data.print_3d_array(grouped_data)

        result_expected = np.zeros((3, 4))
        for sensor in range(grouped_data.shape[2]):
            for row in range(grouped_data.shape[0]):
                result_expected[row, sensor] = np.mean(grouped_data[row, :, sensor])

        result = Mean(sequence=np.nan).aggregate(grouped_data)
        self.assertEqual(result_expected[0, 0], (417 + 147 + 397 + 204 + 417) / 5)
        assert_array_equal(result_expected, result)

    def test_percentile(self):
        #  https://docs.google.com/spreadsheets/d/1KoBUzJf4TIX5xlHIPg4BK6zDAugQWLJ7Lm_lOg2dcLg/edit#gid=0
        result_expected = np.array([
            [204, 539, 27, 302],
            [687, 835, 85, 316],
            [212, 147, 280, 294]
        ])
        result = Percentile(
            percentile=.25,
            interpolation='higher',
            sequence=np.nan).aggregate(grouped_data=helper_data.generated_3d_data())
        assert_array_equal(result_expected, result)

    def test_outliers(self):
        limits = [
            {'input': 'a', 'min': 200, 'max': 500},
            {'input': 'b', 'min': 200, 'max': 600},
            {'input': 'c', 'min': np.nan, 'max': 400},
            {'input': 'd', 'min': 300, 'max': np.nan},
        ]
        # https://docs.google.com/spreadsheets/d/1KoBUzJf4TIX5xlHIPg4BK6zDAugQWLJ7Lm_lOg2dcLg/edit#gid=120567734
        result_expected = np.array([
            [1, 3, 1, 1],
            [5, 4, 1, 1],
            [3, 3, 3, 2]
        ])

        result = Outlier(sequence=np.nan, generate=limits).aggregate(grouped_data=helper_data.generated_3d_data())
        assert_array_equal(result_expected, result)

    def test_trend(self):
        data = np.ma.zeros((2, 3, 2))
        data[:, :, 0] = np.array([
            [1, 2, 3],
            [2, 3, 7]
        ])
        data[:, :, 1] = np.array([
            [11, 12, 13],
            [12, 13, 3],
        ])
        data.mask = np.ma.zeros((2, 3, 2))

        result_expected = np.array([
            [2, 2],
            [5, -9]
        ])
        result = Trend(sequence=np.nan).aggregate(grouped_data=data)
        assert_array_equal(result_expected, result)

    def test_sum(self):
        data = np.zeros((2, 3, 2), dtype='int')
        data[:, :, 0] = np.array([
            [1, 2, 3],
            [2, 3, 7]
        ])
        data[:, :, 1] = np.array([
            [11, 12, 13],
            [12, 13, 3],
        ])

        result_expected = np.array([
            [6, 36],
            [12, 28]
        ])
        result = Sum(sequence=np.nan).aggregate(grouped_data=data)
        assert_array_equal(result_expected, result)
