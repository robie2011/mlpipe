import unittest
import numpy as np
from aggregators import NanCounter, Max, Min, Mean, Percentile, Outlier
from numpy.testing import assert_array_equal
import helpers.data as helper_data
from aggregators.aggregator_input import AggregatorInput
from aggregators.aggregator_output import AggregatorOutput

# data = helper_data.generated_3d_data()
# helpers.print_3d_array
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
        input_data = AggregatorInput(data=data)

        result_expected = np.zeros((3, 4))
        result_expected[0, 0] = 1
        result_expected[1, 1] = 2
        result_expected[0:2, 2] = 2

        result = NanCounter().aggregate(input_data)
        assert_array_equal(result_expected, result.metrics)

    def test_max(self):
        input_data = AggregatorInput(data=helper_data.generated_3d_data())
        result_expected = np.array([
            [417, 878, 419, 685],
            [958, 968, 692, 750],
            [989, 748, 909, 789]
        ])

        result = Max().aggregate(input_data)
        assert_array_equal(result_expected, result.metrics)

    def test_min(self):
        input_data = AggregatorInput(data=helper_data.generated_3d_data())
        result_expected = np.array([
            147, 92, 0, 198,
            170, 533, 18, 39,
            103, 130, 19, 53]).reshape(3, 4)

        result = Min().aggregate(input_data)
        assert_array_equal(result_expected, result.metrics)

    def test_mean(self):
        input_data = AggregatorInput(data=helper_data.generated_3d_data())
        result_expected = np.zeros((3, 4))
        for sensor in range(input_data.data.shape[2]):
            for row in range(input_data.data.shape[0]):
                result_expected[row, sensor] = np.mean(input_data.data[row, :, sensor])

        result = Mean().aggregate(input_data)
        assert_array_equal(result_expected, result.metrics)

    def test_percentile(self):
        input_data = AggregatorInput(data=helper_data.generated_3d_data())
        #  https://docs.google.com/spreadsheets/d/1KoBUzJf4TIX5xlHIPg4BK6zDAugQWLJ7Lm_lOg2dcLg/edit#gid=0
        result_expected = np.array([
            [204, 539, 27, 302],
            [687, 835, 85, 316],
            [212, 147, 280, 294]
        ])
        result = Percentile(percentile=.25, interpolation='higher').aggregate(input_data)
        assert_array_equal(result_expected, result.metrics)

    def test_outliers(self):
        input_data = AggregatorInput(helper_data.generated_3d_data())
        limits = [
            {'min': 200, 'max': 500},
            {'min': 200, 'max': 600},
            {'min': 100, 'max': 400},
            {'min': 300, 'max': np.nan},
        ]
        # https://docs.google.com/spreadsheets/d/1KoBUzJf4TIX5xlHIPg4BK6zDAugQWLJ7Lm_lOg2dcLg/edit#gid=120567734
        result_expected = np.array([
            [1, 3, 3, 1],
            [5, 4, 4, 1],
            [3, 3, 4, 2]
        ])

        result = Outlier(limits=limits).aggregate(input_data)
        assert_array_equal(result_expected, result.metrics)
