import unittest
import numpy as np
from numpy.testing import assert_array_equal
from helpers.data import print_3d_array
from groupers import create_sequence_offset_matrix
from datetime import datetime, timedelta
from groupers import GroupInput, HourGrouper, MonthGrouper, YearGrouper, WeekdayGrouper, DayGrouper

sequences = np.array([
            [23.0, 10],
            [23.0, 10],
            [23.3, 11],
            [23.1, 10],

            [23.0, 11],
            [23.0, 12],
            [23.0, 13],
            [23.0, 14],

            [23.0, 14],
            [24.0, 14],
            [25.0, 14],
            [25.0, 14],

            [24.0, 14],
            [25.0, 14],
            [26.0, 14],
            [26.0, 14],

            [27.0, 14],
            [27.0, 14],
            [27.0, 14],
            [28.0, 14]
        ])

sequences.flags.writeable = False


class TestSimpleGroupers(unittest.TestCase):
    def test_hour_grouper(self):
        raw_data = np.zeros((20, 3), dtype='object')
        raw_data[:, 1:] = sequences

        _stamps = np.arange(datetime(2019, 7, 1), datetime(2019, 7, 2), timedelta(minutes=15)).astype(datetime)
        raw_data[:, 0] = _stamps[:raw_data.shape[0]]
        result = HourGrouper().group(GroupInput(raw_data=raw_data, timestamps=raw_data[:, 0]))
        result_expected = np.array([
            0, 0, 0, 0,
            1, 1, 1, 1,
            2, 2, 2, 2,
            3, 3, 3, 3,
            4, 4, 4, 4
        ])

        assert_array_equal(result_expected, result)

    def test_day_grouper(self):
        result = DayGrouper().group(GroupInput(raw_data=None, timestamps=np.arange(
            datetime(2019, 7, 1),
            datetime(2019, 7, 5),
            timedelta(days=1)
        )))

        result_expected = np.arange(1, 5)
        assert_array_equal(result_expected, result)

    def test_month_grouper(self):
        result = MonthGrouper().group(GroupInput(raw_data=None, timestamps=np.array([
            datetime(2019, 7, 1, 12, 0),
            datetime(2019, 8, 1, 12, 0),
            datetime(2019, 9, 1, 12, 0),
            datetime(2019, 10, 1, 12, 0),
            datetime(2019, 11, 1, 12, 0),
        ])))

        result_expected = np.arange(7, 12)
        assert_array_equal(result_expected, result)

    def test_year_grouper(self):
        result = YearGrouper().group(GroupInput(raw_data=None, timestamps=np.array([
            datetime(2019, 7, 1, 12, 0),
            datetime(2020, 8, 1, 12, 0),
            datetime(2021, 9, 1, 12, 0),
            datetime(2022, 10, 1, 12, 0),
            datetime(2023, 11, 1, 12, 0),
        ])))

        result_expected = np.arange(2019, 2024)
        assert_array_equal(result_expected, result)

    def test_weekday_grouper(self):
        result = WeekdayGrouper().group(GroupInput(raw_data=None, timestamps=np.array([
            datetime(2019, 7, 1, 12, 0),
            datetime(2019, 7, 2, 12, 0),
            datetime(2019, 7, 3, 12, 0),
            datetime(2019, 7, 8, 12, 0),
            datetime(2019, 7, 9, 12, 0),
        ])))

        result_expected = np.array([0, 1, 2, 0, 1])
        assert_array_equal(result_expected, result)

